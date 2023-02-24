import json
import pandas as pd
data_path = "data/ORConvQA/"

data = []
with open(data_path + 'train.txt') as f:
    for line in f:
        data.append(json.loads(line))
        
input_qa = []
output_q = []
count = 0
for sub_data in data:
    if len(sub_data['history']) > 1:
        input_text = []
        output_text = ''
        for idx, qa in enumerate(sub_data['history']):
            if idx == len(sub_data['history']) - 1:
                input_text.append('[MASK]')
                input_text.append(qa['answer']['text'])
                output_text = qa['question']
            else:
                input_text.append(qa['question'])
                input_text.append(qa['answer']['text'])
        output_q.append(output_text)
        input_qa.append('[SEP]'.join(input_text))
    else:
        count += 1
data_df = pd.DataFrame({'input': input_qa, 'output': output_q})

data_df.head()
data_df.to_csv("processed_orconvqa_train.csv")


data = []
with open(data_path + 'dev.txt') as f:
    for line in f:
        data.append(json.loads(line))

input_qa = []
output_q = []
count = 0
for sub_data in data:
    if len(sub_data['history']) > 1:
        input_text = []
        output_text = ''
        for idx, qa in enumerate(sub_data['history']):
            if idx == len(sub_data['history']) - 1:
                input_text.append('[MASK]')
                input_text.append(qa['answer']['text'])
                output_text = qa['question']
            else:
                input_text.append(qa['question'])
                input_text.append(qa['answer']['text'])
        output_q.append(output_text)
        input_qa.append('[SEP]'.join(input_text))
    else:
        count += 1
data_df = pd.DataFrame({'input': input_qa, 'output': output_q})

data_df.head()
data_df.to_csv("processed_orconvqa_test.csv")