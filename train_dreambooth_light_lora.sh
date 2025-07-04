export MODEL_NAME="/mnt/md0/realistic-vision-v40"
export INSTANCE_DIR="data"
export OUTPUT_DIR="output"


CUDA_VISIBLE_DEVICES=1 \
accelerate launch --mixed_precision="fp16" train_dreambooth_light_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="A [V] face" \
  --resolution=512  \
  --train_batch_size=1 \
  --num_train_epochs=301 --checkpointing_steps=500 \
  --learning_rate=1e-3 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --cfg_drop_rate 0.1 \
  --rank=1 \
  --down_dim=160 \
  --up_dim=80 \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=5 \
  --validation_prompt="A [V] face" \
  --validation_epochs=300 \
  --train_text_encoder
#  --patch_mlp \
#  --resume_from_checkpoint "latest" \
#   --seed=42



