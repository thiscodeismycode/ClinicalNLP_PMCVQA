export CUDA_VISIBLE_DEVICES=7

python3 test.py \
  --ckp "/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/adj_lr/checkpoint-768" \
  --lora_rank 8 \
  # --ckp "/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/VQA_lora_PMC_LLaMA_PMCCLIP/blank/checkpoint-1382" \
  # --ckp "/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/Finetune_PMC/checkpoint-380" \
