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
requests = []
new_examples = []
queries = []
for i, ex in enumerate(examples):
    if ex['domain'] == 'algorithm' and ex['task_type']=='p2ps':
        domain = ex['domain']
        instance = ex['instance']
        query = ex['user_query']
        if query not in queries:
            queries.append(query)
            pos = ex['positive_document']

            prompt = f"""You are an expert for generating hard negative document for information retrieval tasks. You will be provided with a user query (a coding problem) and a positive document that offers a solution to a different problem but provides the {domain} that is helpful for solving the query.

    **user query**
    {query}

    **positive document**
    {pos}

    Your task is to create a hard negative document.
    - The hard negative document should contain a coding problem using the similar context as the user query but test a different {domain}.
    - The hard negative document should also contain the python solution code.
    - The hard negative document should not be related to algorithm: {instance}.
    Directly response the content of hard negative document.
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
            ex['prompt'] = prompt
            new_examples.append(ex)

print(len(requests))
# Write requests to a JSONL file
file_path = 'hard_negative_input.jsonl'
with open(file_path, 'w', encoding='utf-8') as f:
    for request in requests:
        f.write(json.dumps(request) + '\n')

print(f"File saved to {file_path}")

with open('new_examples.pkl', 'wb') as f:
    pickle.dump(new_examples, f)

