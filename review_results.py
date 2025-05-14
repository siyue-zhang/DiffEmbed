import json
import numpy as np
import re


subset = 'leetcode'
predictions = "/home/siyue/Projects/diffusion_embedder/results/BRIGHT/BrightLeetcode/Dream_emb-TheoremAug/BrightLeetcode_leetcode_predictions.json"
# predictions2 = "/home/siyue/Projects/diffusion_embedder/results/BRIGHT/BrightLeetcode/e5-mistral-7b-instruct/BrightLeetcode_leetcode_predictions.json"
predictions2 = "/home/siyue/Projects/diffusion_embedder/results/BRIGHT/BrightLeetcode/e5-mistral-7b-instruct/BrightLeetcode_leetcode_predictions.json"

# subset = 'theoremqa_questions'
# predictions = "/home/siyue/Projects/diffusion_embedder/results/BRIGHT/BrightTheoremqaQuestions/Dream_emb-TheoremAug/BrightTheoremqaQuestions_theoremqa_questions_predictions.json"
# predictions2 = "/home/siyue/Projects/diffusion_embedder/results/BRIGHT/BrightTheoremqaQuestions/e5-mistral-7b-instruct/BrightTheoremqaQuestions_theoremqa_questions_predictions.json"


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

predictions = load_json(predictions)
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


topk = 10
dream_at_least_one_gold = 0
e5_at_least_one_gold = 0
dream_e5_at_least_one_gold = 0
dream_only = 0  # Dream found gold but E5 didn't
e5_only = 0    # E5 found gold but Dream didn't
for example in data_examples:
    query_id = example['id']
    query = example['query']
    golds = example['gold_ids']

    # Process Dream predictions
    pred = predictions[query_id]
    pred = sort_dict_by_value_desc(pred)
    top_doc_ids = list(pred.keys())[:topk]
    dream = {i: (i in golds) for i in top_doc_ids}
    dream_found_gold = any(dream.values())

    # Process E5 predictions
    pred = predictions2[query_id]
    pred = sort_dict_by_value_desc(pred)
    top_doc_ids = list(pred.keys())[:topk]
    e5 = {i: (i in golds) for i in top_doc_ids}
    e5_found_gold = any(e5.values())

    # Update counters
    if dream_found_gold:
        dream_at_least_one_gold += 1
    if e5_found_gold:
        e5_at_least_one_gold += 1
    
    # Count exclusive findings
    if dream_found_gold and not e5_found_gold:
        dream_only += 1
        
        # # Print query and document information
        # print("\n=== Case where Dream found gold but E5 didn't ===")
        # print(f"Query: {query}")
        # dream_gold_ids = [k for k, v in dream.items() if v]
        # print(f"Dream found gold document(s): {dream_gold_ids}")
        # print("Gold document content:")
        # for gold_id in dream_gold_ids:
        #     print(f"- {doc_lookup[gold_id]}...")  # Show first 300 chars
        # print("\nE5 top prediction:")
        # e5_top_id = list(predictions2[query_id].keys())[0]
        # print(f"- {doc_lookup[e5_top_id]}...")
        # print("=" * 80)
        
        # input()

    if e5_found_gold and not dream_found_gold:
        e5_only += 1

        # Print query and document information
        print("\n=== Case where E5 found gold but Dream didn't ===")
        print(f"Query: {query}")
        e5_gold_ids = [k for k, v in e5.items() if v]
        print(f"E5 found gold document(s): {e5_gold_ids}")
        print("Gold document content:")
        for gold_id in e5_gold_ids:
            print(f"- {doc_lookup[gold_id]}...")  # Show first 300 chars
        print("\nDream top prediction:")
        dream_top_id = list(predictions[query_id].keys())[0]
        print(f"- {doc_lookup[dream_top_id]}...")
        print("=" * 80)
        
        input()

    # Check shared correct predictions
    dream_correct_ids = {k for k, v in dream.items() if v}
    e5_correct_ids = {k for k, v in e5.items() if v}
    if dream_correct_ids.intersection(e5_correct_ids):
        dream_e5_at_least_one_gold += 1

# Print results
print(f"\nResults for top-{topk}:")
print(f'Total {len(data_examples)} queries.')
print(f"Dream found at least one gold: {dream_at_least_one_gold}")
print(f"E5 found at least one gold: {e5_at_least_one_gold}")
print(f"Both found at least one same gold: {dream_e5_at_least_one_gold}")
print(f"Dream found gold but E5 didn't: {dream_only}")
print(f"E5 found gold but Dream didn't: {e5_only}")


# for example in data_examples:
#     query_id = example['id']
#     query = example['query']
#     golds = example['gold_ids']
#     print('-'*50)
#     print("query_id:", query_id)
#     print(query, '\n')
#     pred = predictions[query_id]
#     pred = sort_dict_by_value_desc(pred)
#     topk = 5
#     top_doc_ids = list(pred.keys())[:topk]
#     top_docs = [doc_lookup[k] for k in top_doc_ids]
#     counts = range(topk)
#     correct_counts_1 = 0
#     correct_counts_2 = 0

#     for i, d, c in zip(top_doc_ids, top_docs, counts):
#         print('=====')
#         print(f'No.{c+1} is gold: ', i in golds)
#         if i in golds:
#             correct_counts_1 += 1
#         print(d[:min(300,len(d))])
#         print('=====\n')

#     pred = predictions2[query_id]
#     pred = sort_dict_by_value_desc(pred)
#     top_doc_ids = list(pred.keys())[:5]
#     top_docs = [doc_lookup[k] for k in top_doc_ids]
#     counts = range(topk)
#     for i, d, c in zip(top_doc_ids, top_docs, counts):
#         print('#####')
#         print(f'No.{c+1} is gold: ', i in golds)
#         if i in golds:
#             correct_counts_2 += 1
#         print(d[:min(300,len(d))])
#         print('#####\n')

#     print('\n\n')
#     if correct_counts_2<correct_counts_1:
#         break

