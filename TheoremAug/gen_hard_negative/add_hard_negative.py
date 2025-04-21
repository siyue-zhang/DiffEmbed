import json
import re
import pickle
import argparse
from collections import defaultdict, Counter

import sys
sys.path.append('../')
from instances import *

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

file_path = '../output/augmentation_data_leetcode.jsonl'
examples = load_jsonl(file_path)
print('\n', Counter([ex['domain'] for ex in examples if ex['task_type']=='p2ps']),'\n')

with open('new_examples.pkl', 'rb') as f:
    loaded_examples = pickle.load(f)

def extract_code(text):
    pattern = re.compile(r'```(python|java|cpp)?\n(.*?)```', re.DOTALL)
    matches = pattern.findall(text)
    
    if not matches:
        return text, None
    
    return matches[0][1], matches[0][0]

new_hard_negatives = load_jsonl('check_hard_negative_output.jsonl')
example_to_add = []
for i, (example, hard) in enumerate(zip(loaded_examples, new_hard_negatives)):
    # if i < 402:
    #     continue 

    content = hard['response']['body']['choices'][0]['message']['content']
    # print(content)
    content, language = extract_code(content)
    # print(language,'\n')
    if language == 'python':
        example['hard_negative_document'] = content
        example['negative_instance'] = 'unknown'
        example_to_add.append(example)
        # print('\n=============================\n')
        # print(example['user_query'],'\n')
        # print(example['instance'],'\n')
        # print('POS\n',example['positive_document'],'\n')
        # print('NEG\n',content)

print(len(example_to_add),' / ',len(loaded_examples))

examples += example_to_add

file_path = '../output/TheoremAug.jsonl'
with open(file_path, 'w', encoding='utf-8') as f:
    for request in examples:
        f.write(json.dumps(request) + '\n')
print('\n', Counter([ex['domain'] for ex in examples if ex['task_type']=='p2ps']),'\n')
print(f"File saved to {file_path}")

