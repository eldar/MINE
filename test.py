import sys
import os
from pathlib import Path
from collections import namedtuple
import argparse
import logging
import math
import json
import gzip
import pickle
import pdb

from tqdm import tqdm
import numpy as np
from imageio.v3 import imread, imwrite
import cv2
import yaml
import torch
import torch.nn.functional as F

from synthesis_task import SynthesisTask
from operations import mpi_rendering


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


def write_img_to_disk(img_tensor, step, postfix, output_dir):
    B, C, H, W = img_tensor.size()
    assert C==1 or C==3
    img_np_BHWC = img_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    img_np_BHWC_255 = np.clip(np.round(img_np_BHWC * 255), a_min=0, a_max=255)
    for b in range(B):
        if C == 3:
            cv2.imwrite(os.path.join(output_dir, '%d_%d_%s.png'%(step, b, postfix)),
                        cv2.cvtColor(img_np_BHWC_255[b], cv2.COLOR_RGB2BGR))
        elif C == 1:
            cv2.imwrite(os.path.join(output_dir, '%d_%d_%s.png'%(step, b, postfix)),
                        img_np_BHWC_255[b, :, :, 0])


def get_w2c_4x4(pose):
    w2c = np.concatenate((pose.astype(np.float32),
            np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0)
    return w2c


def estimate_depth_scale(depth, sparse_depth):
    """
    depth: [1, 1, H, W]
    sparse_depth: [N, 3]
    """
    eps = 1e-7
    device = depth.device
    sparse_depth = sparse_depth.to(device)
    if sparse_depth.shape[0] < 10:
        return torch.tensor(1.0, device=device, dtype=torch.float32)
    xy = sparse_depth[:, :2]
    z = sparse_depth[:, 2]
    xy = xy.unsqueeze(0).unsqueeze(0)
    depth_pred = F.grid_sample(depth, xy.to(depth.device), align_corners=False)
    depth_pred = depth_pred.squeeze()
    # z = torch.max(z, torch.tensor(eps, dtype=z.dtype, device=z.device))
    good_depth = z > eps
    z = z[good_depth]
    depth_pred = depth_pred[good_depth]

    if z.shape[0] < 10:
        return torch.tensor(1.0, device=device, dtype=torch.float32)

    scale = (depth_pred.log() - z.log()).mean().exp()

    # if torch.any(torch.isnan(depth_pred.log())):
    #     print("pred is NaN")
        # pdb.set_trace()
    # if torch.any(torch.isnan(z.log())):
    #     print("gt is NaN")
        # pdb.set_trace()
        # assert False, "gt is NaN"

    # if torch.any(torch.isnan(scale)):
    #     print("logit", (depth_pred.log() - z.log()).mean())
    #     assert False, "scale is NaN"
    return scale


def load_test_split(mode):
    split_path = Path("configs/input_pipelines/realestate10k/test_data_jsons")
    split_file_name = "test_pairs.json" if mode == "test" else "validation_pairs.json"
    with open(split_path / split_file_name, "r") as f:
        lines = f.readlines()
        split_data = list(map(json.loads, lines))
    return split_data


Frame = namedtuple('Frame', ['seq_id', 'name', 'timestamp', 'image_file', 'K', 'w2c'])

def create_test_set(data_path, split):
    mode = "test"
    dataset_path = Path(data_path)
    samples = []
    for sample in split:
        seq_id = sample["sequence_id"]
        frame_names = [ "src_img_obj", 
                        "tgt_img_obj_5_frames", 
                        "tgt_img_obj_10_frames",
                        "tgt_img_obj_random"]
        frames = []
        for frame_name in frame_names:
            frame = sample[frame_name]
            timestamp = int(frame["frame_ts"])
            image_file = dataset_path / mode / seq_id / f"{timestamp}.jpg"
            pose = np.reshape(np.array(frame["camera_pose"]), (3, 4))
            w2c = np.concatenate(
                (pose.astype(np.float32),
                np.array([[0, 0, 0, 1]], dtype=np.float32)),
                axis=0
            )
            intr = frame["camera_intrinsics"]
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = intr[0]
            K[1, 1] = intr[1]
            K[0, 2] = intr[2]
            K[1, 2] = intr[3]

            frame = Frame(
                seq_id=seq_id,
                name=frame_name,
                timestamp=timestamp,
                image_file=image_file,
                w2c=w2c,
                K=K
            )

            frames.append(frame)
            # id = np.searchsorted(seq_data[key]["timestamps"], timestamp)
            # frame_ids.append(id)
        
        samples.append(frames)
    
    return samples


class Predictor:
    def __init__(self, synthesis_task, config, seq_data, mode, output_dir):
        self.synthesis_task = synthesis_task
        self.config = config

        self.synthesis_task.global_step = config["training.eval_interval"]
        self.synthesis_task.logger.info("Start running evaluation on validation set:")
        self.synthesis_task.backbone.eval()
        self.synthesis_task.decoder.eval()

        self.seq_data = seq_data
        self.mode = mode
        self.data_path = Path(config["data.path"])
        self.output_dir = Path(output_dir)

    def resize(self, img):
        W, H = self.config["data.img_w"], self.config["data.img_h"]
        img = cv2.resize(img, (W, H), cv2.INTER_LINEAR)
        return img

    def traj_generation(self, frames):
        tgts_poses = []
        src_pose = torch.from_numpy(frames[0].w2c)
        for frame in frames[1:]:
            tgt_pose = torch.from_numpy(frame.w2c)
            # TODO is this correct?
            tgt_src = tgt_pose @ torch.linalg.inv(src_pose)
            tgts_poses += [tgt_src]
        return tgts_poses

    def infer_network(self, frames):
        img = imread(frames[0].image_file)
        orig_size = img.shape[:2]
        img = self.resize(img)
        img = torch.from_numpy(img).cuda().permute(2, 0, 1).contiguous().unsqueeze(0) / 255.0

        device = img.device

        B, _, H, W = img.size()
        
        src_frame = frames[0]

        K = torch.from_numpy(src_frame.K.copy()).unsqueeze(0).to(device)
        K[:, 0, :] *= W
        K[:, 1, :] *= H
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
        seq_id = src_frame.seq_id
        pcl_file = self.data_path / "pcl" / self.mode / seq_id / "pcl.pickle"
        with gzip.open(pcl_file, "rb") as f:
            sparse_pcl = pickle.load(f)
        pose_data = self.seq_data[seq_id]
        frame_idx = np.where(pose_data["timestamps"] == src_frame.timestamp)[0].item()
        xyd = self.get_sparse_depth(pose_data, orig_size, sparse_pcl, frame_idx)
        self.scale_factor = estimate_depth_scale(src_depth_syn, xyd)

    @staticmethod
    def get_sparse_depth(pose_data, orig_size, sparse_pcl, frame_idx):
        # image_id-1 == frame_idx
        xys_all = sparse_pcl["xys"]
        p3D_ids_all = sparse_pcl["p3D_ids"]
        xyz = sparse_pcl["xyz"]

        xys = xys_all[frame_idx]
        p3D_ids = p3D_ids_all[frame_idx]

        H, W = orig_size

        visible_points = p3D_ids != -1
        xys = xys[visible_points, :]
        p3D_ids = p3D_ids[visible_points]

        xyz_image = xyz[p3D_ids, :]
        xyz_image_h = np.hstack((xyz_image, np.ones_like(xyz_image[:, :1])))

        # ===== compute point projections onto image with network data ====
        # index to -1 because image_ids are 1-indexed
        # K = _process_projs(pose_data["intrinsics"][image_id-1], H, W)
        # load the extrinsic matrixself.num_scales
        T_w2c = get_w2c_4x4(pose_data["poses"][frame_idx])
        # P = K @ T_w2c
        xyz_pix = np.einsum("ji,ni->nj", T_w2c, xyz_image_h)[:, :3]
        depth = xyz_pix[:, 2:]
        img_dim = np.array([[W, H]])
        xys_scaled = (xys / img_dim - 0.5) * 2
        xyd = np.concatenate([xys_scaled, depth], axis=1)
        return torch.from_numpy(xyd).to(torch.float32)

    def render_views(self, frames, poses):
        src_frame = frames[0]
        scale_factor = 1.0 / self.scale_factor
        # scale_factor = torch.tensor([1.0])

        tgt_img_np_list = []
        tgt_disp_np_list = []

        for frame, pose in zip(frames[1:], poses):
            G_tgt_src = pose.unsqueeze(0).to(self.mpi_all_rgb_src.device)

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

            tgt_img_np = img_tensor_to_np(tgt_imgs_syn)
            tgt_disp_np = img_tensor_to_np(tgt_disparity_syn)
            tgt_img_np_list.append(tgt_img_np)
            tgt_disp_np_list.append(tgt_disp_np)

            imwrite(self.output_dir / f"{frame.seq_id}.{src_frame.timestamp}.{frame.name}.pred.jpg", tgt_img_np, quality=95)

        # save GT images
        for frame in frames:
            imwrite(self.output_dir / f"{frame.seq_id}.{src_frame.timestamp}.{frame.name}.gt.jpg", self.resize(imread(frame.image_file)), quality=95)


def main():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpus", type=str, required=True)
    parser.add_argument("--extra_config", type=str, default="{}", required=False)
    args = parser.parse_args()

    # Enable cudnn benchmark for speed optimization
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.backends.cudnn.benchmark = True

    # Load config yaml file
    extra_config = json.loads(args.extra_config)
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

    mode = "test"
    split = load_test_split(mode)
    samples = create_test_set(args.data_path, split)

    seq_data_path = Path(args.data_path) / f"{mode}.pickle"
    with open(seq_data_path, "rb") as f:
        seq_data = pickle.load(f)

    predictor = Predictor(synthesis_task, config, seq_data, mode, args.output_dir)

    with torch.no_grad():
        for frames in tqdm(samples):
            # load source image
            if not frames[0].image_file.exists():
                continue
            tgts_poses = predictor.traj_generation(frames)
            predictor.infer_network(frames)
            predictor.render_views(frames, tgts_poses)


if __name__ == '__main__':
    main()
