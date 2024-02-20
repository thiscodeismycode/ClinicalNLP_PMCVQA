import argparse
import os
import json
import math
import tqdm.auto as tqdm
from typing import Optional
import transformers
from Dataset.Slake_Dataset import Slake_Dataset
from Dataset.PMC_QA_Dataset import PMC_QA_Dataset
from models.QA_model import QA_model
from transformers import Trainer
from dataclasses import dataclass, field
import os
from torch.utils.data import DataLoader  
import torch
# import wandb


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="chaoyi-wu/PMC_LLAMA_7B")
    ckp: Optional[str] = field(default="/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/VQA_lora_PMC_LLaMA_PMCCLIP/blank/checkpoint-1382")

    ## Q_former ##
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32) 

    ## Basic Setting ##
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096)
    checkpointing: Optional[bool] = field(default=True)

    ## Image Encoder ##
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default='../MedVInT_TE/models/pmc_clip/checkpoint.pt')
    #visual_model_config: Optional[str] = field(default='./img_checkpoint/RN50_fusion4.json')
    
    ## Peft ##
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)


@dataclass
class DataArguments:
    Train_csv_path: str = field(default='/home/user/KHJ/PMC-VQA/PMC-VQA/train.csv', metadata={"help": "Path to the training data."})
    Eval_csv_path: str = field(default='/home/user/KHJ/PMC-VQA/PMC-VQA/valid.csv', metadata={"help": "Path to the training data."})
    tokenizer_path: str = field(default='chaoyi-wu/PMC_LLAMA_7B', metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")

    # Train_dataset = Slake_Dataset(data_args.Train_csv_path, data_args.tokenizer_path, img_dir='/home/user/KHJ/PMC-VQA/PMC-VQA/images/images_train/', text_type = 'blank')
    # Eval_dataset = Slake_Dataset(data_args.Eval_csv_path, data_args.tokenizer_path, img_dir='/home/user/KHJ/PMC-VQA/PMC-VQA/images/images_valid/', text_type = 'blank')
    Train_dataset = PMC_QA_Dataset(csv_path=data_args.Train_csv_path, tokenizer_path=data_args.tokenizer_path,
                                   img_dir='/home/user/KHJ/PMC-VQA/PMC-VQA/images/images_train/', text_type = 'blank')
    Eval_dataset = PMC_QA_Dataset(csv_path=data_args.Eval_csv_path, tokenizer_path=data_args.tokenizer_path,
                                  img_dir='/home/user/KHJ/PMC-VQA/PMC-VQA/images/images_valid/', text_type = 'blank')


    print("Setup Model")
    ckp = model_args.ckp + '/pytorch_model.bin'
    model = QA_model(model_args)
    print("Loading Pre-train Model")

    ckpt = torch.load(ckp, map_location='cpu')
    #if you have problem in loading, it may cause by the peft package updating and use the following code:
    for name in list(ckpt.keys()):
        if 'self_attn.q_proj.weight' in name and "vision_model" not in name:
            new_name = name.replace('self_attn.q_proj.weight', 'self_attn.q_proj.base_layer.weight')
            ckpt[new_name] = ckpt.pop(name)
        if 'self_attn.v_proj.weight' in name and "vision_model" not in name:
            new_name = name.replace('self_attn.v_proj.weight', 'self_attn.v_proj.base_layer.weight')
            ckpt[new_name] = ckpt.pop(name)
        if 'lora_A' in name:
            new_name = name.replace('lora_A', 'lora_A.default')
            ckpt[new_name] = ckpt.pop(name)
        if 'lora_B' in name:
            new_name = name.replace('lora_B', 'lora_B.default')
            ckpt[new_name] = ckpt.pop(name)

    model.load_state_dict(ckpt, strict=False)

    print('Start training')
    trainer = Trainer(model=model, 
                      train_dataset=Train_dataset,
                      eval_dataset=Eval_dataset,
                      args=training_args,
                      )

    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
