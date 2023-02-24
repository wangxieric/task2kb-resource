import json
import pandas as pd

# data_path = "data/MultiWoZ/train/"
# data_path = "data/MultiWoZ/dev/"
data_path = "data/MultiWoZ/test/"

input_qa = []
output_q = []
count = 0
for i in range(1, 3):
    file_name = f'dialogues_{i:03d}.json'
    data = json.load(open(data_path + file_name))
    for sub_data in data:
        count += 1
        context = []
        for idx, turn in enumerate(sub_data['turns']):
            if turn['utterance'].strip()[-1] == '?' and turn['speaker'] == 'USER':
                output_q.append(turn['utterance'])
                input_qa.append('[SEP]'.join(context) + '[SEP][MASK][SEP]' + sub_data['turns'][idx + 1]['utterance'])
                context.append(turn['utterance'])
            else:
                context.append(turn['utterance'])

data_df = pd.DataFrame({'input': input_qa, 'output': output_q})
data_df.head()
data_df.to_csv("processed_multiwoz_test.csv")