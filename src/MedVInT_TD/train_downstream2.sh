export CUDA_VISIBLE_DEVICES=7

wandb enabled
wandb login --relogin a1ff7e1a0641445b3ebffc021e1bd9f0c6a43f67

python3 train_downstream2.py \
    --Train_csv_path '/home/user/KHJ/PMC-VQA/PMC-VQA/train_multi2.csv' \
    --Eval_csv_path '/home/user/KHJ/PMC-VQA/PMC-VQA/valid.csv' \
    --ckp '/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/VQA_lora_PMC_LLaMA_PMCCLIP/blank/checkpoint-1382'\
    --output_dir ./Results/0327 \
    --run_name 0327 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --report_to 'wandb' \
    --lora_rank 4 \
    --vision_output_dir '/home/user/KHJ/PMC-VQA/src/MedVInT_TE/models/pmc_clip/new-ckp.pt' \
    --clip_only False \
    --freeze_clip False \
    --freeze_llama False \
    --seq_length 512 \
    --img_token_num 32 \
    # --model_path 'PharMolix/BioMedGPT-LM-7B' \
    # --tokenizer_path 'PharMolix/BioMedGPT-LM-7B' \
    # --visual_model_path '/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/VQA_lora_PMC_LLaMA_PMCCLIP/checkpoint.pt' \
    # --model_path 'GerMedBERT/medalpaca-7b' \
    # --tokenizer_path 'GerMedBERT/medalpaca-7b' \
    # --img_tokens 128 \
    # --img_token_num 128 \
    # --Train_csv_path '/home/user/KHJ/PMC-VQA/PMC-VQA/train2.csv' \
    # --checkpointing false \
    # --save_steps 10 \
    # --eval_steps 10 \
    # --deepspeed ./ds_config/ds_config_zero2.json \
    # --ckp '/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/VQA_lora_PMC_LLaMA_PMCCLIP/blank/checkpoint-1382'\

wandb disabled
