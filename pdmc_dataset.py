import imp
import json
from lib2to3.pgen2 import token
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
class MDDDatasetAnchor(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        # 获取第一列和第二列的数据
        self.ids = df['Participant_ID'].tolist()
        self.labels = df['PHQ8_Binary'].tolist()
        self.phq8 = df['PHQ8_Score'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        meta = int(self.ids[idx])
        json_file = os.path.join('./daic/daic-json', str(self.ids[idx]) + '_TRANSCRIPT.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
        family_text = data['Relationships with family members']
        work_text = data['Relationship with colleagues at work']
        mental_text = data['Personal mental state']
        medical_text = data['Personal medical history']
        abstract_text = data['Comprehensive evaluation']

        family = self.tokenizer(family_text, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        work = self.tokenizer(work_text, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        mental = self.tokenizer(mental_text, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        medical = self.tokenizer(medical_text, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        abstract = self.tokenizer(abstract_text, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        family_input_ids = family['input_ids'].squeeze(dim=0)
        family_attention_mask = family['attention_mask'].squeeze(dim=0)
        work_input_ids = work['input_ids'].squeeze(dim=0)
        work_attention_mask = work['attention_mask'].squeeze(dim=0)
        mental_input_ids = mental['input_ids'].squeeze(dim=0)
        mental_attention_mask = mental['attention_mask'].squeeze(dim=0)
        medical_input_ids = medical['input_ids'].squeeze(dim=0)
        medical_attention_mask = medical['attention_mask'].squeeze(dim=0)
        abstract_input_ids = abstract['input_ids'].squeeze(dim=0)
        abstract_attention_mask = abstract['attention_mask'].squeeze(dim=0)


        label = torch.tensor(self.labels[idx], dtype=torch.long)
        phq8 = torch.tensor(self.phq8[idx], dtype=torch.long)
        batch = (family_input_ids, family_attention_mask, work_input_ids, work_attention_mask, mental_input_ids, mental_attention_mask, medical_input_ids, medical_attention_mask, abstract_input_ids, abstract_attention_mask, label, phq8)
        return meta, batch


class MDDDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        df = pd.read_csv(csv_file)
        self.ids = df['Participant_ID'].tolist()
        self.labels = df['PHQ8_Binary'].tolist()
        # self.phq8 = df['PHQ8_Score'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        meta = int(self.ids[idx])
        json_file = os.path.join('./daic/daic-json', str(self.ids[idx]) + '_TRANSCRIPT.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
        family_text = data['Relationships with family members']
        work_text = data['Relationship with colleagues at work']
        mental_text = data['Personal mental state']
        medical_text = data['Personal medical history']
        abstract_text = data['Comprehensive evaluation']

        family = self.tokenizer(family_text, add_special_tokens=True, max_length=int(self.max_length/2), padding='max_length', truncation=True, return_tensors='pt')
        work = self.tokenizer(work_text, add_special_tokens=True, max_length=int(self.max_length/2), padding='max_length', truncation=True, return_tensors='pt')
        mental = self.tokenizer(mental_text, add_special_tokens=True, max_length=int(self.max_length/2), padding='max_length', truncation=True, return_tensors='pt')
        medical = self.tokenizer(medical_text, add_special_tokens=True, max_length=int(self.max_length/2), padding='max_length', truncation=True, return_tensors='pt')
        abstract = self.tokenizer(abstract_text, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        family_input_ids = family['input_ids'].squeeze(dim=0)
        family_attention_mask = family['attention_mask'].squeeze(dim=0)
        work_input_ids = work['input_ids'].squeeze(dim=0)
        work_attention_mask = work['attention_mask'].squeeze(dim=0)
        mental_input_ids = mental['input_ids'].squeeze(dim=0)
        mental_attention_mask = mental['attention_mask'].squeeze(dim=0)
        medical_input_ids = medical['input_ids'].squeeze(dim=0)
        medical_attention_mask = medical['attention_mask'].squeeze(dim=0)
        abstract_input_ids = abstract['input_ids'].squeeze(dim=0)
        abstract_attention_mask = abstract['attention_mask'].squeeze(dim=0)


        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # phq8 = torch.tensor(self.phq8[idx], dtype=torch.long)
        batch = (family_input_ids, family_attention_mask, work_input_ids, work_attention_mask, mental_input_ids, mental_attention_mask, medical_input_ids, medical_attention_mask, abstract_input_ids, abstract_attention_mask, label)
        return meta, batch
