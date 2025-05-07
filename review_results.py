import json
import numpy as np
import re

subset = 'leetcode'

def load_json(file_path):
    """Load and return data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{file_path}'.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

predictions = "/home/siyue/Projects/diffusion_embedder/results/BRIGHT/BrightLeetcode/Dream_emb-TheoremAug/BrightLeetcode_leetcode_predictions.json"
predictions = load_json(predictions)

predictions2 = "/home/siyue/Projects/diffusion_embedder/results/BRIGHT/BrightLeetcode/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-TheoremAug/BrightLeetcode_leetcode_predictions.json"
predictions2 = load_json(predictions2)

from datasets import load_dataset

# "examples" is a DatasetDict (train/validation/test)
data_examples = load_dataset("xlangai/BRIGHT", "examples")
data_examples = data_examples[subset]

def sort_dict_by_value_desc(data):
    """Sort a dictionary by its values in descending order."""
    return dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

data_documents = load_dataset("xlangai/BRIGHT", "documents")
data_documents = data_documents[subset]

doc_lookup = {}
for doc in data_documents:
    doc_id = doc["id"]
    doc_text = doc["content"]
    doc_lookup[doc_id] = doc_text
print('loaded doc_lookup')



for example in data_examples:
    query_id = example['id']
    query = example['query']
    golds = example['gold_ids']
    print('-'*50)
    print("query_id:", query_id)
    print(query, '\n')
    pred = predictions[query_id]
    pred = sort_dict_by_value_desc(pred)
    topk = 5
    top_doc_ids = list(pred.keys())[:topk]
    top_docs = [doc_lookup[k] for k in top_doc_ids]
    counts = range(topk)
    correct_counts_1 = 0
    correct_counts_2 = 0

    for i, d, c in zip(top_doc_ids, top_docs, counts):
        print('=====')
        print(f'No.{c+1} is gold: ', i in golds)
        if i in golds:
            correct_counts_1 += 1
        print(d[:min(300,len(d))])
        print('=====\n')

    pred = predictions2[query_id]
    pred = sort_dict_by_value_desc(pred)
    top_doc_ids = list(pred.keys())[:5]
    top_docs = [doc_lookup[k] for k in top_doc_ids]
    counts = range(topk)
    for i, d, c in zip(top_doc_ids, top_docs, counts):
        print('#####')
        print(f'No.{c+1} is gold: ', i in golds)
        if i in golds:
            correct_counts_2 += 1
        print(d[:min(300,len(d))])
        print('#####\n')

    print('\n\n')
    if correct_counts_2<correct_counts_1:
        break

