# export WANDB_DISABLED=true
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=7 \
CUDA_LAUNCH_BLOCKING=1 \
#srun --partition=gpu --mpi=pmi2 --gres=gpu:2 -n1 --ntasks-per-node=1  --job-name=VQA_LoRA_training --kill-on-bad-exit=1 \

#torchrun --nproc_per_node=1 --master_port 18832
python train.py \
    --bf16 True \
    --output_dir ./Results/VQA_lora_pmcclip \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --eval_steps 5 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --run_name VQA_LoRA_training \
    --tf32 True \
    --is_blank True \
    --image_encoder "PMC_CLIP" \
    --pmcclip_pretrained "./models/pmc_clip/checkpoint.pt"
    # --deepspeed ./ds_config/ds_config_zero2.json \ if deep_speed
    # --pretrained_model ./PMC_LLAMA_Model  \ if PMC-LLaMA, change this to your PMC-LLaMA model path
    # --image_encoder "PMC_CLIP" \ if PMC-CLIP

