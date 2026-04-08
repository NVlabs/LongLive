# mr2v inference: multi-reference-to-video generation
# Usage: Edit reference_images in configs/longlive_mr2v_inference.yaml (0-3 image paths)
torchrun \
  --nproc_per_node=8 \
  --master_port=12321 \
  mr2v_inference.py \
  --config_path configs/longlive_mr2v_inference.yaml
