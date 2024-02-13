export PATH=/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=7 \

#torchrun --nproc_per_node=2 --master_port 19934
python3 finetune.py \
    --bf16 True \
    --output_dir ./Results \
    --pretrained_model "chaoyi-wu/PMC_LLAMA_7B"  \
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --run_name PMC-VQA\
    --tf32 True \
    --is_blank True \
    --image_encoder "PMC_CLIP" \
    --pmcclip_pretrained "./models/pmc_clip/checkpoint.pt"
