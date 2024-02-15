import argparse
import os
import json
import math
import tqdm.auto as tqdm
from typing import Optional
import transformers
from Dataset.Slake_Dataset import Slake_Dataset
from models.QA_model import QA_model
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
    
    ## Peft ##
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)


@dataclass
class DataArguments:
    img_dir: str = field(default='/home/user/KHJ/PMC-VQA/PMC-VQA/images/images_valid/', metadata={"help": "Path to validation images."})
    Test_csv_path: str = field(default='/home/user/KHJ/PMC-VQA/PMC-VQA/valid.csv', metadata={"help": "Path to valdiation data."})
    tokenizer_path: str = field(default='chaoyi-wu/PMC_LLAMA_7B', metadata={"help": "Path to the pretrained tokenizer."})
    trier: int = field(default=0)


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
    # if os.path.exists('result_final'+str(data_args.trier)+'.csv'): 
        
    #     with open('result_final'+str(data_args.trier)+'.csv', 'r') as file:
    #         reader = csv.reader(file)
    #         row_count = sum(1 for row in reader)-1      
    Test_dataset_close = Slake_Dataset(data_args.Test_csv_path, data_args.tokenizer_path,mode='Test',text_type='blank',start=row_count)
    
    # batch size should be 1
    Test_dataloader_close = DataLoader(
            Test_dataset_close,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
    ) 
    Test_dataset_open = Slake_Dataset(data_args.Test_csv_path, data_args.tokenizer_path,mode='Test',text_type='blank',start=row_count)
    
    # batch size should be 1
    Test_dataloader_open = DataLoader(
            Test_dataset_open,
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
    print(ckp)
    model = QA_model(model_args)

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
    model = model.to('cuda')
    model.eval()

    output = []

    for sample in tqdm.tqdm(Test_dataloader_open):

        input_ids = Test_dataset_open.tokenizer(sample['input_ids'], return_tensors="pt").to('cuda')
        images = sample['images'].to('cuda')
        with torch.no_grad():
            generation_ids = model.generate_long_sentence(input_ids['input_ids'], images)
        # generated_texts = Test_dataset.tokenizer.batch_decode(generation_ids.argmax(-1), skip_special_tokens=True)
        generated_texts = Test_dataset_open.tokenizer.batch_decode(generation_ids, skip_special_tokens=True)

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
    
    """ACC = 0
    cc = 0
    model = model.to('cuda')
    model.eval()
    #Test_dataset.tokenizer.padding_side = "left" 
    
    with open('result_final_greedy_SLAKE_close'+model_args.ckp.split('/')[-3]+'_'+ model_args.ckp.split('/')[-2]+'.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Figure_path','Question','Pred','Label'])
            for sample in tqdm.tqdm(Test_dataloader_close):
                input_ids = Test_dataset_close.tokenizer(sample['input_ids'],return_tensors="pt").to('cuda')
                #input_ids['input_ids'][0][0]=1
                images = sample['images'].to('cuda')
                with torch.no_grad():
                    generation_ids = model.generate_long_sentence(input_ids['input_ids'],images)
                generated_texts = Test_dataset_close.tokenizer.batch_decode(generation_ids, skip_special_tokens=True) 
                for i in range(len(generated_texts)):
                    label = sample['labels'][i]
                    img_path = sample['img_path'][i]
                    pred = generated_texts[i]
                    writer.writerow([img_path,sample['input_ids'][i],pred,label])
                    # writer.writerow([pred])
                    cc = cc + 1
    
    with open('result_final_greedy_SLAKE_open'+model_args.ckp.split('/')[-3]+'_'+ model_args.ckp.split('/')[-2]+'.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Figure_path','Question','Pred','Label'])
            for sample in tqdm.tqdm(Test_dataloader_open):
                input_ids = Test_dataset_open.tokenizer(sample['input_ids'],return_tensors="pt").to('cuda')
                #input_ids['input_ids'][0][0]=1
                images = sample['images'].to('cuda')
                with torch.no_grad():
                    generation_ids = model.generate_long_sentence(input_ids['input_ids'],images)
                generated_texts = Test_dataset_open.tokenizer.batch_decode(generation_ids, skip_special_tokens=True) 
                for i in range(len(generated_texts)):
                    label = sample['labels'][i]
                    img_path = sample['img_path'][i]
                    pred = generated_texts[i]
                    writer.writerow([img_path, sample['input_ids'][i].encode('utf-8'), pred, label])
                    # writer.writerow([pred])
                    cc = cc + 1"""


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()
