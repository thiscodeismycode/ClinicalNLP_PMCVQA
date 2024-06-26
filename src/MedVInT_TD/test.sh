export CUDA_VISIBLE_DEVICES=7

python3 test.py \
  --ckp "/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/0327/checkpoint-198" \
  --Test_csv_path "/home/user/KHJ/PMC-VQA/PMC-VQA/testset.csv" \
  --visual_model_path '/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/VQA_lora_PMC_LLaMA_PMCCLIP/checkpoint.pt' \
  --lora_rank 4 \
  --no_query False \
  --img_token_num 32 \
  --seq_length 512  \
  # --model_path 'PharMolix/BioMedGPT-LM-7B' \
  # --tokenizer_path 'PharMolix/BioMedGPT-LM-7B' \
  # --visual_model_path '/home/user/KHJ/PMC-VQA/src/MedVInT_TE/models/pmc_clip/checkpoint.pt' \
  # --ckp "/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/Finetune_pretext/checkpoint-528" \
  # --ckp "/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/VQA_lora_PMC_LLaMA_PMCCLIP/blank/checkpoint-1382" \
  # --ckp "/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/Finetune_PMC/checkpoint-380" \
  # /home/user/KHJ/PMC-VQA/PMC-VQA/testset.csv
