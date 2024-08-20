import argparse
import json
from lib2to3.pgen2 import token
import os
import pandas as pd
from torch.utils.data import Dataset
import torch

from transformers import pipeline
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=0)
args = parser.parse_args()

model_id = '/data1/hf_model/llama/llama3/Meta-Llama-3-8B'
model_id = '/data1/hf_model/llama/llama3/Meta-Llama-3-8B-Instruct'
model_id = '/data1/hf_model/llama/llama2/Llama-2-7b-chat-hf'

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

test_csv_file = './daic/dev_split_Depression_AVEC2017.csv'
df = pd.read_csv(test_csv_file)
ids = df['Participant_ID'].tolist()
labels = df['PHQ8_Binary'].tolist()
phq8s = df['PHQ8_Score'].tolist()
print(ids)
print(phq8s)
print(labels)

preds = []

print(len(ids), len(phq8s), len(labels))
for idx in range(len(ids)):
    json_file = os.path.join('./daic/daic-json', str(ids[idx]) + '_TRANSCRIPT.json')
    with open(json_file, 'r') as f:
        data = json.load(f)
    family_text = data['Relationships with family members']
    work_text = data['Relationship with colleagues at work']
    mental_text = data['Personal mental state']
    medical_text = data['Personal medical history']
    abstract_text = data['Comprehensive evaluation']
    label = labels[idx]
    phq8 = phq8s[idx]
    # label = torch.tensor(labels[idx], dtype=torch.long)
    # phq8 = torch.tensor(phq8s[idx], dtype=torch.long)

    print(50*'*', idx, str(ids[idx]), 50*'*')
    print(label, phq8)
    print(family_text)
    messages_all = [
        {"role": "system", "content": "You're a mental health diagnostic assistant!"},
        {"role": "user", "content": f"Given you a person's family background, work situation, mental state, medical history, and overall, please help me diagnose and output <depression> or <not depression>. Family background: {family_text},  Work situation: {work_text}, Mental state: {mental_text}, Medical history: {medical_text}, Overall: {abstract_text}"},
    ]

    wo_family_messages = [
        {"role": "system", "content": "You're a mental health diagnostic assistant!"},
        {"role": "user", "content": f"Given you a person's work situation, mental state, medical history, and overall, please help me diagnose and output <depression> or <not depression>. Work situation: {work_text}, Mental state: {mental_text}, Medical history: {medical_text}, Overall: {abstract_text}"},
    ]

    wo_work_messages = [
        {"role": "system", "content": "You're a mental health diagnostic assistant!"},
        {"role": "user", "content": f"Given you a person's family background, mental state, medical history, and overall, please help me diagnose and output <depression> or <not depression>. Family background: {family_text}, Mental state: {mental_text}, Medical history: {medical_text}, Overall: {abstract_text}"},
    ]

    wo_mental_messages = [
        {"role": "system", "content": "You're a mental health diagnostic assistant!"},
        {"role": "user", "content": f"Given you a person's family background, work situation, medical history, and overall, please help me diagnose and output <depression> or <not depression>. Family background: {family_text}, Work situation: {work_text}, Medical history: {medical_text}, Overall: {abstract_text}"},
    ]

    wo_medical_messages = [
        {"role": "system", "content": "You're a mental health diagnostic assistant!"},
        {"role": "user", "content": f"Given you a person's family background, work situation, mental state, and overall, please help me diagnose and output <depression> or <not depression>. Family background: {family_text},  Work situation: {work_text}, Mental state: {mental_text}, Overall: {abstract_text}"},
    ]

    wo_overall_messages = [
        {"role": "system", "content": "You're a mental health diagnostic assistant!"},
        {"role": "user", "content": f"Given you a person's family background, work situation, mental state, medical history, please help me diagnose and output <depression> or <not depression>. Family background: {family_text},  Work situation: {work_text}, Mental state: {mental_text}, Medical history: {medical_text}"},
    ]
    
    # log_name =  
    if args.index == 0:
        messages = messages_all
    elif args.index == 1:
        messages = wo_family_messages
    elif args.index == 2:
        messages = wo_work_messages
    elif args.index == 3:
        messages = wo_mental_messages
    elif args.index == 4:
        messages = wo_medical_messages
    elif args.index == 5:
        messages = wo_overall_messages

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    assistant_response = outputs[0]["generated_text"][-1]["content"]
    print(assistant_response)
    
    temp = -1
    if '<depression>' in assistant_response:
        preds.append(1)
        temp = 1
    elif '<not depression>' in assistant_response:
        preds.append(0)
        temp = 0
    else:
        preds.append(2)
        temp = 2
    print(f'pred:{temp}')

print('labels = ', labels)
print('preds  = ', preds)







