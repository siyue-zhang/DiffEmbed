import json
import numpy as np
from datasets import load_dataset

# Load the first file (TheoremAug_tmp.jsonl)
theorem_aug_data = []
with open('../output/TheoremAug_tmp.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        theorem_aug_data.append(json.loads(line.strip()))

# Load the second file (gen_hard_negative_input.jsonl)
hard_negative_data = []
with open('gen_hard_negative_output.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        hard_negative_data.append(json.loads(line.strip()))

# Print the number of items loaded from each file
print(f"Loaded {len(theorem_aug_data)} items from TheoremAug_tmp.jsonl")
print(f"Loaded {len(hard_negative_data)} items from gen_hard_negative_input.jsonl")

responses = [data['response']['body']['choices'][0]['message']['content'] for data in hard_negative_data]
response_by_id = {data['custom_id']: data['response']['body']['choices'][0]['message']['content'] for data in hard_negative_data}

# TOODO : check if the hard_negative_document uses the instance
finals = []
for example in theorem_aug_data:
    if example['hard_negative_document'] == None:
        idx = example['id']
        example['hard_negative_document'] = response_by_id[idx]
    finals.append(example)

# TOODO : remove the questions too similar to the BRIGHT test questions
bright_queries = []
data_examples = load_dataset("xlangai/BRIGHT", "examples")
for subset in ['aops', 'leetcode', 'theoremqa_theorems', 'theoremqa_questions']:
    bright_queries += [data['query'] for data in data_examples[subset]]

def jaccard_similarity_qs(q1, q2):
    # Tokenize by splitting on spaces and lowercase for normalization
    set1 = set(q1.lower().split())
    set2 = set(q2.lower().split())
    
    # Avoid division by zero
    if not set1 or not set2:
        return 0.0
    
    # Jaccard formula: intersection over union
    intersection = set1 & set2
    union = set1 | set2
    
    return len(intersection) / len(union)

all_max_scores = []
max_indices = []
for example in theorem_aug_data:
    query = example['user_query']
    scores = [jaccard_similarity_qs(query, q) for q in bright_queries]
    m = max(scores)
    all_max_scores.append(m)
    max_indices.append(scores.index(m))
    if m > 0.5:
        print(f"Query: {query}")
        print(f"Max Jaccard Similarity: {m}")
        print(f"Bright Queries: {[bright_queries[i] for i in range(len(bright_queries)) if scores[i] == m]}\n")

max_aug_score = max(all_max_scores)
max_aug_index = all_max_scores.index(max_aug_score)
query = theorem_aug_data[max_aug_index]['user_query']
jac = all_max_scores[max_aug_index]
bright = bright_queries[max_indices[max_aug_index]]
print(f"Max Jaccard Similarity: {np.round(jac,3)}")
print(f"Query: {query}\n")
print(f"Bright Query: {bright}\n")
print(len(bright_queries))

# Save the updated data to TheoremAug.jsonl
with open('../output/TheoremAug.jsonl', 'w', encoding='utf-8') as f:
    for item in finals:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')