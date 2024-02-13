import argparse
import os
import csv
import json
import math
import numpy as np
import tqdm.auto as tqdm
from typing import Optional
import difflib 
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
from torch import nn
from torch.utils.data import DataLoader  
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F

from dataset.dataset import Binary_VQA_Dataset
from models.llama.vqa_model import Binary_VQA_Model

"""
Output should include the following information:

- encounter id (of the input query)
- responses
    - content_zh: ""
    - content_en: "some response"
    - content_es: ""
    
* content in zh and es should be left blank.
Wouldn't it be better if we change this code to write json file directly,\
so that we don't have to write another code that changes csv to json?
Ya let's do this!
"""


@dataclass
class ModelArguments:
    embed_dim: Optional[int] = field(default=768)
    pretrained_tokenizer:  Optional[str] = field(default="chaoyi-wu/PMC_LLAMA_7B")
    pretrained_model: Optional[str] = field(default="chaoyi-wu/PMC_LLAMA_7B")
    image_encoder: Optional[str] = field(default="PMC_CLIP")
    # image_encoder: Optional[str] = field(default="PMC_CLIP")
    pmcclip_pretrained: Optional[str] = field(default="./models/pmc_clip/checkpoint.pt")
    clip_pretrained: Optional[str] = field(default="openai/clip-vit-base-patch32")
    # ckp: Optional[str] = field(default="./Results/VQA_lora_pmcclip/vqa/checkpoint-13500")
    # ckp: Optional[str] = field(default="./Results/vqa/checkpoint-3800")
    ckp: Optional[str] = field(default="./Results/My_VQA_lora_pmcclip/vqa/checkpoint-3800")


@dataclass
class DataArguments:
    is_blank: Optional[bool] = field(default=True)
    image_res: Optional[int] = field(default=512)
    img_root_dir: str = field(default='../../PMC-VQA/images/images_valid', metadata={"help": "Path to validation data."})
    Test_csv_path: str = field(default='../../PMC-VQA/valid.csv', metadata={"help": "Path to the validation data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    logging_dir: Optional[str] = field(default="./logs")
    logging_steps: Optional[int] = field(default=50)


def get_generated_texts(label, outputs, tokenizer):
    outputs = outputs[label != 0][1:-1]
    generated_text = tokenizer.decode(outputs)
    return generated_text


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")
    Test_dataset = Binary_VQA_Dataset(data_args.Test_csv_path, data_args.img_root_dir, data_args.image_res,
                                      is_blank=data_args.is_blank, is_train=False)
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
    model = Binary_VQA_Model(model_args)
    model.load_state_dict(torch.load(ckp), strict=False)

    print("Start Testing")
    
    model = model.to('cuda')
    model.eval()

    output = []

    for sample in tqdm.tqdm(Test_dataloader):
        new_item = {}
        img_path = sample['image_path']
        encounter_id = sample['encounter_id']
        question = sample['question']
        image = sample['image'].to('cuda')
        label = sample['label'].to('cuda')[:, 0, :]
        question_inputids = sample['encoded_input_ids'].to('cuda')[:, 0, :]
        question_attenmask = sample['encoded_attention_mask'].to('cuda')[:, 0, :]

        with torch.no_grad():
            outputs = model(image, question_inputids, question_attenmask)
        loss = F.nll_loss(outputs.transpose(1, 2), label, ignore_index=0)

        generated_texts = get_generated_texts(label, outputs.argmax(-1), Test_dataset.tokenizer)

        new_item["encounter_id"] = encounter_id
        new_item["query_content"] = question
        new_item["response"] = {
            "content_zh": "",
            "content_en": generated_texts,
            "content_es": ""
        }

        output.append(new_item)

    with open(os.path.join(training_args.output_dir, 'prediction.json'), mode='w') as outfile:
        json.dump(output, outfile)


if __name__ == "__main__":
    main()
    
