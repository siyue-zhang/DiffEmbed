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

with open('new_examples.pkl', 'rb') as f:
    loaded_examples = pickle.load(f)


new_hard_negatives = load_jsonl('hard_negative_output.jsonl')
requests=[]
for i, (example, hard) in enumerate(zip(loaded_examples, new_hard_negatives)):
    content = hard['response']['body']['choices'][0]['message']['content']
    instance = example['instance']
    prompt = f"""{content}

Is this document about the {instance}? If yes, response YES. If no, postprocess this document, keep only the solution code and write the question description in the docstring of the code. 
"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        ]
    requests.append({
        "custom_id": str(i), 
        "method": "POST", 
        "url": "/v1/chat/completions", 
        "body": 
            {"model": "gpt-4o-mini", 
                "messages": messages,
            "max_tokens": 2000},
    })

print(len(requests))
# Write requests to a JSONL file
file_path = 'check_hard_negative_input.jsonl'
with open(file_path, 'w', encoding='utf-8') as f:
    for request in requests:
        f.write(json.dumps(request) + '\n')
