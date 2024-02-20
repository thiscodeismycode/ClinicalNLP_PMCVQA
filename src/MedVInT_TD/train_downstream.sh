export CUDA_VISIBLE_DEVICES=7 \
# torchrun --nproc_per_node=2 --master_port 18340 train_downstream.py \


python3 train_downstream.py \
    --Train_csv_path '/home/user/KHJ/PMC-VQA/PMC-VQA/valid.csv' \
    --Eval_csv_path '/home/user/KHJ/PMC-VQA/PMC-VQA/valid.csv' \
    --output_dir ./Results/insanityy \
    --run_name insanityy \
    --num_train_epochs 2 \
    --eval_steps 50 \
    --save_steps 50 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --checkpointing false \
    --bf16 True \
    --tf32 True \
    # --deepspeed ./ds_config/ds_config_zero2.json \

