from transformers import AutoModelForCausalLM, AutoTokenizer
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



device = "cuda" # the device to load the model onto

model_id = '/data1/hf_model/qwen/Qwen-14B-Chat'
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,)




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
        {"role": "system", "content": "You're a mental health diagnostic assistant! You must answer from the options:<depression, not depression>!"},
        {"role": "user", "content": f"Given you a person's family background, work situation, mental state, medical history, and overall, please help me diagnose and must output the format <depression> or <not depression>. Family background: {family_text},  Work situation: {work_text}, Mental state: {mental_text}, Medical history: {medical_text}, Overall: {abstract_text}"},
    ]

    wo_family_messages = [
        {"role": "system", "content": "You're a mental health diagnostic assistant! You must answer from the options:<depression, not depression>!"},
        {"role": "user", "content": f"Given you a person's work situation, mental state, medical history, and overall, please help me diagnose and must output the format <depression> or <not depression>. Work situation: {work_text}, Mental state: {mental_text}, Medical history: {medical_text}, Overall: {abstract_text}"},
    ]

    wo_work_messages = [
        {"role": "system", "content": "You're a mental health diagnostic assistant! You must answer from the options:<depression, not depression>!"},
        {"role": "user", "content": f"Given you a person's family background, mental state, medical history, and overall, please help me diagnose and must output the format <depression> or <not depression>. Family background: {family_text}, Mental state: {mental_text}, Medical history: {medical_text}, Overall: {abstract_text}"},
    ]

    wo_mental_messages = [
        {"role": "system", "content": "You're a mental health diagnostic assistant! You must answer from the options:<depression, not depression>!"},
        {"role": "user", "content": f"Given you a person's family background, work situation, medical history, and overall, please help me diagnose and must output the format <depression> or <not depression>. Family background: {family_text}, Work situation: {work_text}, Medical history: {medical_text}, Overall: {abstract_text}"},
    ]

    wo_medical_messages = [
        {"role": "system", "content": "You're a mental health diagnostic assistant! You must answer from the options:<depression, not depression>!"},
        {"role": "user", "content": f"Given you a person's family background, work situation, mental state, and overall, please help me diagnose and must output the format <depression> or <not depression>. Family background: {family_text},  Work situation: {work_text}, Mental state: {mental_text}, Overall: {abstract_text}"},
    ]

    wo_overall_messages = [
        {"role": "system", "content": "You're a mental health diagnostic assistant! You must answer from the options:<depression, not depression>!"},
        {"role": "user", "content": f"Given you a person's family background, work situation, mental state, medical history, please help me diagnose and must output the format <depression> or <not depression>. Family background: {family_text},  Work situation: {work_text}, Mental state: {mental_text}, Medical history: {medical_text}"},
    ]
    
    # log_name =  
    if args.index == 0:
        prompt = messages_all
    elif args.index == 1:
        prompt = wo_family_messages
    elif args.index == 2:
        prompt = wo_work_messages
    elif args.index == 3:
        prompt = wo_mental_messages
    elif args.index == 4:
        prompt = wo_medical_messages
    elif args.index == 5:
        prompt = wo_overall_messages




    # messages = [
    #     {"role": "system", "content": "You're a mental health diagnostic assistant!"},
    #     {"role": "user", "content": prompt}
    # ]
    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    temp = -1
    if '<depression>' in response:
        preds.append(1)
        temp = 1
    elif '<not depression>' in response:
        preds.append(0)
        temp = 0
    else:
        preds.append(2)
        temp = 2
    print(f'pred:{temp}')


print('labels = ', labels)
print('preds  = ', preds)
