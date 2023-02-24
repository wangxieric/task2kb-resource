# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
import random
import sys
from datetime import datetime
# Importing the T5 modules from huggingface/transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import T5Tokenizer, AutoConfig, AutoModelForSeq2SeqLM

import pickle
import gzip
import ast

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'  

def load_first_input(df_dir):
    # data = pd.read_csv(df_dir, engine="python", error_bad_lines=False)
    data = pd.read_csv(df_dir)
    dialogue_inputs = {}
    full_dialogue = {}
    selected_items = data.loc[data.index.isin(data.drop_duplicates(subset='dialogue_id').index)]
    rest_data = data.loc[~data.index.isin(data.drop_duplicates(subset='dialogue_id').index)]
    for index, row in selected_items.iterrows():
        task = "How to " + ' '.join(row['urls'][24:].split('-')) + '?'
        intro = ast.literal_eval(row['task_info'])['intro']
        text = task + '[SEP]' + intro + '[SEP][MASK][SEP]' + row['response']
        # 1/2/n qa with no intro 
        dialogue_inputs[row['urls'] + '-' + str(row['dialogue_id'])] = task + '[SEP][MASK][SEP]' + row['response']
        # 1/2/n qa with intro
        # dialogue_inputs[row['urls'] + str(row['dialogue_id'])] = text
        full_dialogue[row['urls'] + '-' + str(row['dialogue_id'])] = text
    return dialogue_inputs, full_dialogue, rest_data

def load_rest_input(dialogue_inputs, full_dialogue, data):
    selected_items = data.loc[data.index.isin(data.drop_duplicates(subset='dialogue_id').index)]
    rest_data = data.loc[~data.index.isin(data.drop_duplicates(subset='dialogue_id').index)]
    for index, row in selected_items.iterrows():
        # task = ' '.join(row['urls'][24:].split('-'))
        task = "How to " + ' '.join(row['urls'][24:].split('-')) + '?'
        intro = ast.literal_eval(row['task_info'])['intro']
        
        # 1qa + intro
        # text = task + '[SEP]' + intro + '[SEP][MASK][SEP]' + row['reponse']
        
        # 1qa + no intro
        # text = task + '[SEP][MASK][SEP]' + row['reponse']
        # dialogue_inputs[row['urls'] + '-' + str(row['dialogue_id'])] = text
        
        full_dialogue[row['urls'] + '-' + str(row['dialogue_id'])] = full_dialogue[row['urls'] + '-' + str(row['dialogue_id'])] + '[SEP][MASK][SEP]' + row['response']
        # 2qa
        text = full_dialogue[row['urls'] + '-' + str(row['dialogue_id'])].split('[SEP]')
        dialogue_inputs[row['urls'] + '-' + str(row['dialogue_id'])] = text[0] + '[SEP]' + '[SEP]'.join(text[-4:])
        # nqa = full dialogue
        # dialogue_inputs[row['urls'] + '-' + str(row['dialogue_id'])] = text[0] + '[SEP]' + '[SEP]'.join(text[2:])
        
    return dialogue_inputs, full_dialogue, rest_data


class CustomDataset(Dataset):

    def __init__(self, data_input, tokenizer, input_len, output_len):
        self.tokenizer = tokenizer
        self.data = pd.DataFrame.from_dict(data_input, orient='index', columns=['text'])
        self.input_len = input_len
        self.output_len = output_len
        self.source = self.data.text
        # actually the urls are concatenated with dialogue id         
        self.urls = list(self.data.index)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source = str(self.source[index])
        source = ' '.join(source.split())
        
        source_tok = self.tokenizer.batch_encode_plus([source], truncation=True, max_length= self.input_len, pad_to_max_length=True,return_tensors='pt')
        
        source_ids = source_tok['input_ids'].squeeze()
        source_mask = source_tok['attention_mask'].squeeze()
        
        task_urls = self.urls[index]
        return {
            'task_urls': task_urls,
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
        }
    
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model_path = 'saved_model/Flan_T5_large/MultiWoZ/'
PATH = model_path + 'saved_model'
model.load_state_dict(torch.load(PATH)['model_state_dict'])
model = model.to(device)

max_len = 512
output_len = 200
BATCH_SIZE = 8
params = {
        'batch_size': BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

data_types = ['art', 'car', 'computer', 'education', 'family', 'finance', 'food', 'health', 'hobby', 'holiday', 'home', 'pet', 'philosophy', 'relationship', 'sport', 'style', 'travel', 'work', 'youth']
# with open('generation_process_output_2.txt', 'w') as f:
#     sys.stdout = f
for category in data_types:
    print("Start Processing Category: ", category)
    print("First Run. ", datetime.now())
    DATA_PATH = f'data/Task2KB/data_for_dialogue_gen_3s/{category}_for_convqa_first_three_sent.csv'
    dialogue_inputs, full_dialogue, rest_data = load_first_input(DATA_PATH)
    data_set = CustomDataset(dialogue_inputs, tokenizer, max_len, output_len)
    data_loader = DataLoader(data_set, **params)
    with torch.no_grad():
        for _, data in enumerate(data_loader, 0):
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            task_urls = data['task_urls']
            generated_ids = model.generate(
                    input_ids = ids,
                    attention_mask = mask, 
                    max_length=max_len, 
                    num_beams=2,
                    repetition_penalty=2.5, 
                    length_penalty=1.0, 
                    early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            for idx, url in enumerate(task_urls):
                # dialogue_inputs[url] = dialogue_inputs[url].replace('[MASK]', preds[idx])
                full_dialogue[url] = full_dialogue[url].replace('[MASK]', preds[idx])
                # print(dialogue_inputs[url])

    iterates = 1
    while len(rest_data) != 0:
        print(str(iterates) + " iterated_run: ", datetime.now())
        print("The number of remaining data to process: ", len(rest_data))
        dialogue_inputs, full_dialogue, rest_data = load_rest_input(dialogue_inputs, full_dialogue, rest_data)
        data_set = CustomDataset(dialogue_inputs, tokenizer, max_len, output_len)
        data_loader = DataLoader(data_set, **params)
        with torch.no_grad():
            for _, data in enumerate(data_loader, 0):
                ids = data['source_ids'].to(device, dtype = torch.long)
                mask = data['source_mask'].to(device, dtype = torch.long)
                task_urls = data['task_urls']
                generated_ids = model.generate(
                        input_ids = ids,
                        attention_mask = mask, 
                        max_length=max_len, 
                        num_beams=2,
                        repetition_penalty=2.5, 
                        length_penalty=1.0, 
                        early_stopping=True
                        )
                preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                for idx, url in enumerate(task_urls):
                    # dialogue_inputs[url] = dialogue_inputs[url].replace('[MASK]', preds[idx])
                    full_dialogue[url] = full_dialogue[url].replace('[MASK]', preds[idx])

        iterates += 1
        if iterates > 5:
            break

    for index, row in rest_data.iterrows():
        full_dialogue.pop(row['urls'] + str(row['dialogue_id']), None)

    print("number of dialogues: ", len(full_dialogue.keys()))
    print("full_dialogue", full_dialogue)
    with open(f"{category}_multiwoz_convqa_flan_t5_large_2qa_3sent.json", "w") as outfile:
        json.dump(full_dialogue, outfile)