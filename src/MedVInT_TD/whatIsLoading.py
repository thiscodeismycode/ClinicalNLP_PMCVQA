import torch

ckp1 = '/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/VQA_lora_PMC_LLaMA_PMCCLIP/blank/checkpoint-1382/pytorch_model.bin'
ckp2 = '/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/Finetune_PMC/checkpoint-53/pytorch_model.bin'
ckpt1 = torch.load(ckp1, map_location='cpu')
ckpt2 = torch.load(ckp2, map_location='cpu')

list1 = ckpt1.keys()
list2 = ckpt2.keys()
list_out = [key for key in list2 if key not in list1]

for key in list_out:
    print(key)
