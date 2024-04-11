python visualizations/image_to_video.py \
  --data_path visualizations/home.jpg \
  --checkpoint_path checkpoints/MINE_realestate10k_384x256_monodepth2_N32/checkpoint.pth \
  --gpus 1 --output_dir tmp

python test.py \
  --data_path visualizations/home.jpg \
  --checkpoint_path checkpoints/MINE_realestate10k_384x256_monodepth2_N32/checkpoint.pth \
  --gpus 1 --output_dir tmp
