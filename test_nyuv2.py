import sys
import os
from pathlib import Path
from collections import namedtuple
import logging
import json
import pickle
import pdb

from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
import numpy as np
from imageio.v3 import imread, imwrite
import cv2
import yaml
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from synthesis_task import SynthesisTask
from operations import mpi_rendering

from datasets.nyu.dataset import NYUv2Dataset
from datasets.collate import custom_collate
from util.depth import \
    estimate_depth_scale, \
    estimate_depth_scale_ransac


def disparity_normalization_vis(disparity):
    """
    :param disparity: Bx1xHxW, pytorch tensor of float32
    :return:
    """
    assert len(disparity.size()) == 4 and disparity.size(1) == 1
    disp_min = torch.amin(disparity, (1, 2, 3), keepdim=True)
    disp_max = torch.amax(disparity, (1, 2, 3), keepdim=True)
    disparity_syn_scaled = (disparity - disp_min) / (disp_max - disp_min)
    disparity_syn_scaled = torch.clip(disparity_syn_scaled, 0.0, 1.0)
    return disparity_syn_scaled


def img_tensor_to_np(img_tensor):
    B, C, H, W = img_tensor.size()
    assert B == 1
    assert C == 1 or C == 3
    img_np_HWC = img_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()[0]
    img_np_HWC_255 = np.clip(np.round(img_np_HWC * 255), a_min=0, a_max=255).astype(np.uint8)
    if C == 1:
        img_np_HWC_255 = cv2.applyColorMap(img_np_HWC_255, cv2.COLORMAP_HOT)
        img_np_HWC_255 = cv2.cvtColor(img_np_HWC_255, cv2.COLOR_BGR2RGB)
    return img_np_HWC_255


class Predictor:
    def __init__(self, synthesis_task, config, output_dir):
        self.synthesis_task = synthesis_task
        self.config = config

        self.synthesis_task.global_step = config["training.eval_interval"]
        self.synthesis_task.logger.info("Start running evaluation on validation set:")
        self.synthesis_task.backbone.eval()
        self.synthesis_task.decoder.eval()

        self.data_path = Path(config["data.path"])
        self.output_dir = Path(output_dir)

    def traj_generation(self, frames):
        tgts_poses = []
        src_pose = torch.from_numpy(frames[0].w2c)
        for frame in frames[1:]:
            tgt_pose = torch.from_numpy(frame.w2c)
            # TODO is this correct?
            tgt_src = tgt_pose @ torch.linalg.inv(src_pose)
            tgts_poses += [tgt_src]
        return tgts_poses

    def resize(self, img):
        W, H = self.config["data.img_w"], self.config["data.img_h"]
        img = TF.resize(img, (H, W))
        return img

    def infer_network(self, inputs):
        input_f_id = 0

        K = inputs[("K_src", input_f_id)]
        img = inputs["color", input_f_id, 0]
    
        orig_size = img.shape[2:]
        H_orig, W_orig = orig_size

        img = self.resize(img)
        img = img.cuda()

        device = img.device

        B, _, H, W = img.size()
        
        K[:, 0, :] *= W / W_orig
        K[:, 1, :] *= H / H_orig
        K_inv = torch.inverse(K).to(device)
        self.K, self.K_inv = K, K_inv

        N_pt = 128

        src_items = {
            "img": img,
            "K": self.K,
            "K_inv": self.K_inv,
            "xyzs": torch.ones((B, 3, N_pt), dtype=torch.float32)
        }
        tgt_items = {
            "img": img.unsqueeze(1),
            "K": self.K.unsqueeze(1),
            "K_inv": self.K_inv.unsqueeze(1),
            "xyzs": torch.ones((B, 1, 3, N_pt), dtype=torch.float32),
            "G_src_tgt": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        }
        self.synthesis_task.set_data((src_items, tgt_items))

        # self.xyz_src_BS3HW
        endpoints = self.synthesis_task.network_forward()
        self.disparity_all_src = endpoints["disparity_all_src"]
        mpi_all_src = endpoints["mpi_all_src_list"][0]

        # Do RGB blending
        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
            self.synthesis_task.homography_sampler_list[0].meshgrid,
            self.disparity_all_src,
            self.K_inv.to(img.device)
        )
        self.mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
        self.mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
        src_imgs_syn, src_depth_syn, blend_weights, weights = mpi_rendering.render(
            self.mpi_all_rgb_src,
            self.mpi_all_sigma_src,
            xyz_src_BS3HW,
            use_alpha=self.config.get("mpi.use_alpha", False),
            is_bg_depth_inf=self.config.get("mpi.render_tgt_rgb_depth", False)
        )
        self.mpi_all_rgb_src = blend_weights * img.unsqueeze(1) + (1 - blend_weights) * self.mpi_all_rgb_src

        # get sparse depth map from COLMAP
        sparse_depth = inputs[("depth_sparse", input_f_id)][0]
        self.scale_factor = estimate_depth_scale_ransac(src_depth_syn, sparse_depth)

    def render_views(self, inputs):
        scale_factor = 1.0 / self.scale_factor

        input_f_id = 0
        target_frame_ids = [1, 2, 3]

        tgt_img_np_list = []
        tgt_disp_np_list = []

        src_pose = inputs[("T_w2c", input_f_id)]

        novel_frame_names = [
            "tgt_img_obj_5_frames",
            "tgt_img_obj_10_frames",
            "tgt_img_obj_random"
        ]

        src_frame_id = inputs[("frame_id", input_f_id)][0]

        for f_id, frame_name in zip(target_frame_ids, novel_frame_names):
            tgt_pose = inputs[("T_w2c", f_id)]
            tgt_src = tgt_pose @ torch.linalg.inv(src_pose)

            G_tgt_src = tgt_src.to(self.mpi_all_rgb_src.device)

            render_results = self.synthesis_task.render_novel_view(
                self.mpi_all_rgb_src,
                self.mpi_all_sigma_src,
                self.disparity_all_src, G_tgt_src,
                self.K_inv, self.K,
                scale=0,
                scale_factor=scale_factor.to(G_tgt_src.device)
            )
            tgt_imgs_syn = render_results["tgt_imgs_syn"]
            tgt_disparity_syn = render_results["tgt_disparity_syn"]
            tgt_disparity_syn = disparity_normalization_vis(tgt_disparity_syn)

            tgt_img_gt = inputs[("color", f_id, 0)]
            # tgt_img_gt = self.resize(tgt_img_gt)
            tgt_img_gt_np = img_tensor_to_np(tgt_img_gt)
            imwrite(self.output_dir / f"{src_frame_id}.{frame_name}.gt.png", tgt_img_gt_np)

            H, W = tgt_img_gt_np.shape[:2]
            tgt_imgs_syn = TF.resize(tgt_imgs_syn, (H, W))
            tgt_img_np = img_tensor_to_np(tgt_imgs_syn)
            tgt_disp_np = img_tensor_to_np(tgt_disparity_syn)
            tgt_img_np_list.append(tgt_img_np)
            tgt_disp_np_list.append(tgt_disp_np)

            imwrite(self.output_dir / f"{src_frame_id}.{frame_name}.pred.png", tgt_img_np)


class ArgParserStub:
    def __init__(self):
        self.checkpoint_path = "/work/eldar/src/MINE/checkpoints/MINE_realestate10k_384x256_monodepth2_N32/checkpoint.pth"
        self.data_path = "/scratch/shared/nfs1/eldar/data/realestate10k"
        self.gpus = "1"
        self.output_dir = "/work/eldar/src/MINE/output/nyuv2_stan"


def create_loader(cfg, dataset):
    B = 1 #cfg.optimiser.batch_size
    dataloader = DataLoader(
        dataset, B, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True, drop_last=False,
        collate_fn=custom_collate
    )
    return dataloader


@hydra.main(config_path="configs_hydra", config_name="config", version_base=None)
def main(cfg : DictConfig) -> None:
    # os.chdir(Path(__file__).parents[0])
    os.chdir("/work/eldar/src/MINE")
    print("CWD:", os.getcwd())

    args = ArgParserStub()

    # Enable cudnn benchmark for speed optimization
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.backends.cudnn.benchmark = True

    # Load config yaml file
    extra_config = dict()
    config_path = os.path.join(os.path.dirname(args.checkpoint_path), "params.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f)
        for k in extra_config.keys():
            assert k in config, k
        config.update(extra_config)

    # preprocess config
    config["current_epoch"] = 0
    config["global_rank"] = 0
    config["training.pretrained_checkpoint_path"] = args.checkpoint_path

    # pre-process params
    config["mpi.disparity_list"] = np.zeros((1), dtype=np.float32)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        config["local_workspace"] = args.output_dir

    # logging to file and stdout
    config["log_file"] = os.path.join(args.output_dir, "inference.log")
    logger = logging.getLogger("graph_view_synthesis_inference")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
    stream_handler.setFormatter(formatter)
    logger.handlers = [stream_handler]
    logger.propagate = False

    config["data.path"] = args.data_path
    config["logger"] = logger
    config["tb_writer"] = None  # SummaryWriter(args.output_dir)
    config["data.per_gpu_batch_size"] = 1

    synthesis_task = SynthesisTask(config=config, logger=logger, is_val=True)

    predictor = Predictor(synthesis_task, config, args.output_dir)

    nyuv2_dataset = NYUv2Dataset(cfg)
    dataloader = create_loader(cfg, nyuv2_dataset)

    with torch.no_grad():
        for inputs in tqdm(dataloader):
            predictor.infer_network(inputs)
            predictor.render_views(inputs)


if __name__ == '__main__':
    main()
