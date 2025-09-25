torchrun \
  --nproc_per_node=1 \
  --master_port=29500 \
  interactive_inference.py \
  --config_path configs/longlive_interactive_inference.yaml