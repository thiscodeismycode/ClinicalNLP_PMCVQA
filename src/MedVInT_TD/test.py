import argparse
import os
import json
import math
import tqdm.auto as tqdm
from typing import Optional
import transformers
from Dataset.PMC_QA_Dataset import PMC_QA_Dataset
from Dataset.PMC_QA_Dataset_Title import PMC_QA_Dataset_Title
from Dataset.Slake_Dataset import Slake_Dataset
from models.QA_model_CLIP import QA_model_CLIP
from transformers import Trainer
from dataclasses import dataclass, field
import os
from torch.utils.data import DataLoader  
import torch
import numpy as np  
import difflib 
import csv


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="chaoyi-wu/PMC_LLAMA_7B")
    ckp: Optional[str] = field(default="/home/user/KHJ/PMC-VQA/src/MedVInT_TD/Results/VQA_lora_PMC_LLaMA_PMCCLIP/blank/checkpoint-1382")
    checkpointing: Optional[bool] = field(default=False)
    ## Q_former ##
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32)
    
    ## Basic Setting ##
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096)
    
    ## Image Encoder ##
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default='../MedVInT_TE/models/pmc_clip/checkpoint.pt')
    vision_output_dir: Optional[str] = field(default='/home/user/KHJ/PMC-VQA/src/MedVInT_TE/models/pmc_clip/new-ckp.pt')

    ## Peft ##
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

    ## Into the unknown ##
    freeze_clip: Optional[bool] = field(default=True)
    freeze_llama: Optional[bool] = field(default=True)


@dataclass
class DataArguments:
    img_dir: str = field(default='/home/user/KHJ/PMC-VQA/PMC-VQA/images/images_valid/', metadata={"help": "Path to validation images."})
    Test_csv_path: str = field(default='/home/user/KHJ/PMC-VQA/PMC-VQA/valid.csv', metadata={"help": "Path to valdiation data."})
    tokenizer_path: str = field(default='chaoyi-wu/PMC_LLAMA_7B', metadata={"help": "Path to the pretrained tokenizer."})
    trier: int = field(default=0)
    clip_only: Optional[bool] = field(default=False)
    no_query: Optional[bool] = field(default=False)
    seq_length: Optional[int] = field(default=512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()


def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")
    row_count = 0
    Test_dataset = PMC_QA_Dataset_Title(data_args.img_dir, data_args.Test_csv_path, data_args.tokenizer_path,
                                  text_type='blank', mode='Test', start=row_count, clip_only=data_args.clip_only,
                                  no_query=data_args.no_query, seq_length=data_args.seq_length)
    # Test_dataset = Slake_Dataset(img_dir=data_args.img_dir, csv_path=data_args.Test_csv_path,
    #                              tokenizer_path=data_args.tokenizer_path, text_type='blank', mode='Test',
    #                              start=row_count)

    # batch size should be 1
    Test_dataloader = DataLoader(
            Test_dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
    )  

    print("Setup Model")
    ckp = model_args.ckp + '/pytorch_model.bin'
    model = QA_model_CLIP(model_args)
    
    ckpt = torch.load(ckp, map_location='cpu')
    # if you have problem in loading, it may cause by the peft package updating and use the following code:
    """for name in list(ckpt.keys()):
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
            ckpt[new_name] = ckpt.pop(name)"""

    model.load_state_dict(ckpt, strict=False)

    print("Start Testing")

    model = model.to('cuda')
    model.eval()

    output = []

    for sample in tqdm.tqdm(Test_dataloader):

        input_ids = Test_dataset.tokenizer(sample['input_ids'], return_tensors="pt").to('cuda')
        images = sample['images'].to('cuda')
        with torch.no_grad():
            generation_ids = model.generate_long_sentence(input_ids['input_ids'], images)
        # generated_texts = Test_dataset.tokenizer.batch_decode(generation_ids.argmax(-1), skip_special_tokens=True)
        generated_texts = Test_dataset.tokenizer.batch_decode(generation_ids, skip_special_tokens=True)

        for i in range(len(generated_texts)):

            new_item = {}

            encounter_id = sample['encounter_id'][i]
            question = sample['question'][i]
            pred = generated_texts[i]

            new_item["encounter_id"] = encounter_id
            new_item["query_content"] = question
            new_item["responses"] = [{
                "content_zh": "",
                "content_en": pred,
                "content_es": ""
            }]

            output.append(new_item)

    with open(os.path.join(training_args.output_dir, 'prediction.json'), mode='w') as outfile:
        json.dump(output, outfile)

      
if __name__ == "__main__":
    main()
