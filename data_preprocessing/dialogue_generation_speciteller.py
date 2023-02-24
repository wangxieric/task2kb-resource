from models.speciteller.speciteller import getFeatures
from models.speciteller.speciteller import predict
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import re

data_types = ['art', 'car', 'computer', 'education', 'family', 'finance', 'food', 'health', 'hobby', 'holiday', 'home', 'pet', 'philosophy', 'relationship', 'sport', 'style', 'travel', 'work', 'youth']

for data_type in data_types:
    data = json.load(open("data/Task2KB/initial_data/" + data_type + "_articles.json"))
    task_urls = data.keys()
    inputs = []
    text_l_len = []
    selected_urls = []
    task_infos = []
    responses = []
    dialogue_id = 0
    dialogue_ids = []
    for task in tqdm(task_urls):
        # select a sentence from step description as per text specificity
        task_info = data[task]
        method_info = task_info['scripts'][1]['step']
        for method in method_info:
            steps = method['itemListElement']
            if len(steps) > 1:
                for step in steps:
                    # print("1: ", step['text'])
                    text_l = ' '.join(re.split(r'[\r\n]+', step['text']))
                    text_l = text_l.split('. ')
                    # print("3: ", text_l)
                    text_l_len.append(len(text_l))
                    responses.extend(text_l)  
                    selected_urls.append(task)
                    task_infos.append(task_info)
                    dialogue_ids.append(dialogue_id)
            dialogue_id += 1
    print("number of responses: ", len(responses))
    print("start calculating scores, ", datetime.now())
    y,xs,xw = getFeatures(responses)
    preds_comb, preds_s, preds_w = predict(y,xs,xw)
    print("finish calculating scores: ", datetime.now())
    
    selected_responses = []
    idx = 0
    print(len(text_l_len))
    for leng in tqdm(text_l_len):
        begin_idx = sum(text_l_len[:idx])
        end_idx = begin_idx + leng
        if end_idx > len(preds_comb):
            break
        scores = preds_comb[begin_idx:end_idx]
        selected_text = responses[begin_idx:end_idx][np.argmax(scores)]
        selected_responses.append(selected_text)
        idx += 1
    
    print("number of selected_urls: ", len(selected_urls))
    print("number of task_info: ", len(task_infos))
    print("number of selected_responses: ", len(selected_responses))
    print("number of dialogue_ids: ", len(dialogue_ids))
    df = pd.DataFrame(list(zip(selected_urls, dialogue_ids, task_infos, selected_responses)),
               columns =['urls', 'dialogue_id', 'task_info', 'response'])
    
    print("saving " + data_type + " data")
    print(df.shape)
    print(df.head())
    df.to_csv(data_type + '_for_convqa_spec.csv', index = False, encoding='utf-8')