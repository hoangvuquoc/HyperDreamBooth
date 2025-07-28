export MODEL_NAME="/mnt/md0/realistic-vision-v40/"
export PRE_OPTNET_WEIGHT_DIR="output/preoptnet"
export OUTPUT_DIR="output/preoptnet_weight"

CUDA_VISIBLE_DEVICES=2 \
python "export_preoptnet_weight.py" \
  --pretrained_model_name_or_path $MODEL_NAME \
  --pre_opt_weight_path $PRE_OPTNET_WEIGHT_DIR \
  --output_dir $OUTPUT_DIR \
  --rank 1 \
  --down_dim 160 \
  --up_dim 80 \
  --train_text_encoder \
  --total_identities 100 \
  --reference_image_id 10
  # --vit_model_name vit_huge_patch14_clip_336 \
