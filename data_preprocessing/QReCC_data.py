import json
import pandas as pd
data_path = "data/QReCC/"
# train_data
train_data = json.load(open(data_path + 'qrecc_train.json'))
# test data
test_data = json.load(open(data_path + 'qrecc_train.json'))

# process train data
data = train_data

input_qa = []
output_q = []
for qa in data:
    if len(qa['Context']) > 1:
        input_text = qa['Context']
        output_q.append(input_text[-2])
        input_text[-2] = '[MASK]'
        input_text = '[SEP]'.join(input_text)
        input_qa.append(input_text)

data_df = pd.DataFrame({'input': input_qa, 'output': output_q})


data_df.head()
data_df.to_csv("processed_qrecc_train.csv")

#  process test data

data = test_data

input_qa = []
output_q = []
for qa in data:
    if len(qa['Context']) > 1:
        input_text = qa['Context']
        output_q.append(input_text[-2])
        input_text[-2] = '[MASK]'
        input_text = '[SEP]'.join(input_text)
        input_qa.append(input_text)

data_df = pd.DataFrame({'input': input_qa, 'output': output_q})


data_df.head()
data_df.to_csv("processed_qrecc_test.csv")