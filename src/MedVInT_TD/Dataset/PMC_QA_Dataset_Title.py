import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import PIL
import numpy as np
import torch.nn.functional as F
import transformers
import pandas as pd
import random
import copy
from .randaugment import RandomAugment    
from PIL import Image
import tqdm
import csv


class PMC_QA_Dataset_Title(Dataset):
    def __init__(self,  img_dir, csv_path, tokenizer_path, img_tokens=32, seq_length=512, voc_size=32000,
                 mode='Train', start=0, text_type='blank', no_image=False, clip_only=False, no_query=False):
        self.clip_only = clip_only
        self.no_query = no_query
        self.img_root = img_dir
        self.img_root_2 = '/data/clinical-nlp/DermAtlas/'
        self.img_root_3 = '/home/user/KHJ/PMC-VQA/PMC-VQA/images/images_valid/'
        self.data = pd.read_csv(csv_path, delimiter='@', quoting=csv.QUOTE_NONE).iloc[start:]
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token_id = 0
        # self.tokenizer.bos_token_id = 0
        self.tokenizer.eos_token_id = 1
        # self.tokenizer.bos_token_id = 1
        # self.tokenizer.eos_token_id = 2
        self.img_padding = [-100 for i in range(img_tokens)]
        self.attn_padding = [1 for i in range(img_tokens)]
        self.H = 512
        self.W = 512
        self.C = 3
        self.text_type = text_type
        self.no_image = no_image
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([
                transforms.Resize((self.H, self.W)),
                # transforms.RandomResizedCrop((self.H, self.W), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness',
                                                      'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
                                                      'Rotate']),
                transforms.ToTensor(),
                normalize,
            ]) 
        if mode == 'Test':
            self.transform = transforms.Compose([                        
                transforms.Resize((self.H,self.W)),
                transforms.ToTensor(),
                normalize,
            ])
            
        self.mode = mode
        self.seq_length = seq_length
        self.voc_size = voc_size
    
    def random_answer(self, Question, Answer):
        Question = str(Question)
        Answer = str(Answer)
        pre_text = 'Question: '+ Question +'The Answer is:'
        final_o = 'Question: '+ Question +'The Answer is:' + Answer
        return pre_text, final_o

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        encounter_id = sample['Encounter_id']
        if self.clip_only or self.no_query:
            Question = "What is this disease? Please look at the picture."
        else:
            Question = sample['Question']
        Answer = sample['Answer']
        
        if not self.no_image:
        ##### read image pathes #####
            if encounter_id == 0:
                img_path = self.img_root_3 + sample['Figure_path']
            elif len(encounter_id) < 8:  # From our new dataset
                img_path = self.img_root_2 + sample['Figure_path']
            elif len(encounter_id) == 8:
                img_path = self.img_root + sample['Figure_path']
            elif len(encounter_id) > 8:
                img_path = self.img_root_3 + sample['Figure_path']
            img = PIL.Image.open(img_path).convert('RGB') 
            image = self.transform(img) 
        
        if self.mode == 'Train':
            pre_text, final_o = self.random_answer(Question, Answer)
            
            final_o = self.tokenizer(final_o)
            input_ids = final_o['input_ids']
            input_ids.append(self.tokenizer.eos_token_id)
            input_ids = np.array(input_ids)    # Tokenized QA

            input_length = len(input_ids)

            if input_length < self.seq_length:
                input_ids = np.pad(input_ids, (0, self.seq_length - input_length), 'constant', constant_values=0)
            else:
                input_ids = np.append(input_ids[:self.seq_length-1], input_ids[-1])
                # input_ids = input_ids[:self.seq_length]

            label = copy.deepcopy(input_ids)    # Now label is the tokenized QA
            label[label == 0] = -100
            if pre_text != '':
                pre_text = self.tokenizer(pre_text)  # Tokenized Q
                pre_text = pre_text['input_ids']
                if len(pre_text) < len(label):
                    label[:len(pre_text)] = -100   # Mask except for the A part

            label = label.tolist()
            if not self.no_image:
                label = np.array(self.img_padding + label)  # (544,)

                item = {
                    'input_ids': input_ids,
                    'images': image,
                    'labels': label,
                    'encounter_id': encounter_id,
                    'question': Question,
                }
            else:
                label = np.array(label)
                item = {
                    'input_ids': input_ids,
                    'labels': label,
                    'encounter_id': encounter_id,
                    'question': Question,
                }
            return item
        
        if self.mode == 'Test':
            if not self.no_image:
                item = {
                    'input_ids': 'Question: '+ Question + 'The Answer is:',
                    'img_path': sample['Figure_path'],
                    'images': image,
                    'labels': Answer,
                    'encounter_id': encounter_id,
                    'question': Question,
                }
            else:
                item = {
                    'input_ids': 'Question: '+ Question + 'The Answer is:',
                    'img_path': sample['Figure_path'],
                    'labels': Answer,
                    'encounter_id': encounter_id,
                    'question': Question,
                }
            return item
