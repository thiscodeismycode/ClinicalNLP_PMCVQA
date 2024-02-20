export CUDA_VISIBLE_DEVICES=7

wandb enabled
wandb login --relogin a1ff7e1a0641445b3ebffc021e1bd9f0c6a43f67

python3 train_downstream2.py \
    --Train_csv_path '/home/user/KHJ/PMC-VQA/PMC-VQA/train2.csv' \
    --Eval_csv_path '/home/user/KHJ/PMC-VQA/PMC-VQA/valid.csv' \
    --ckp '/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/VQA_lora_PMC_LLaMA_PMCCLIP/blank/checkpoint-1382'\
    --output_dir ./Results/Finetune_PMC \
    --run_name Finetune_PMC \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --report_to 'wandb' \
    # --Train_csv_path '/home/user/KHJ/PMC-VQA/PMC-VQA/train2.csv' \
    # --checkpointing false \
    # --save_steps 10 \
    # --eval_steps 10 \
    # --deepspeed ./ds_config/ds_config_zero2.json \
    # --ckp '/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/VQA_lora_PMC_LLaMA_PMCCLIP/blank/checkpoint-1382'\

wandb disabled
