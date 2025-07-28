# run train
export MODEL_NAME="/mnt/md0/realistic-vision-v40"
export INSTANCE_DIR="CelebA-HQ-1k"
export OUTPUT_DIR="output/preoptnet"

CUDA_VISIBLE_DEVICES=2 \
accelerate launch --mixed_precision="fp16" train_preoptnet.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --instance_data_dir $INSTANCE_DIR \
  --instance_prompt "A [V] face" \
  --output_dir $OUTPUT_DIR \
  --resolution 512 \
  --learning_rate 1e-3 \
  --lr_scheduler constant \
  --checkpoints_total_limit 2 \
  --checkpointing_steps 4 \
  --cfg_drop_rate 0.1 \
  --seed=42 \
  --rank 1 \
  --down_dim 160 \
  --up_dim 80 \
  --train_batch_size 8 \
  --train_steps_per_identity 300 \
  --resume_from_checkpoint "latest" \
  --validation_prompt="A [V] face" \
  --train_text_encoder


# when train_text_encoder, not pre_compute_text_embeddings
#  --pre_compute_text_embeddings \
#  --train_text_encoder
#  --patch_mlp \
