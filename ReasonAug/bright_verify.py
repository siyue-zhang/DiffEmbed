from datasets import load_dataset
from difflib import SequenceMatcher
import json

all_query = []
for subset in ['leetcode','aops','theoremqa_theorems','theoremqa_questions']:
    data_examples = load_dataset("xlangai/BRIGHT", "examples")
    data_examples = data_examples[subset]
    all_query.extend(data_examples['query'])
all_query = list(set(all_query))
print(len(all_query))

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

theorem_aug = load_jsonl("/home/siyue/Projects/diffusion_embedder/TheoremAug/TheoremAug.jsonl")
all_aug_query = [x['user_query'] for x in theorem_aug]
all_aug_query = list(set(all_aug_query))
print(len(all_aug_query))


def check_content_overlap(query, reference_list, threshold=0.7):
    """
    Check if query has high content overlap with any string in reference_list
    Args:
        query: String to check
        reference_list: List of reference strings
        threshold: Similarity threshold (0-1)
    Returns:
        (bool, float): Tuple of (has_overlap, highest_similarity_score)
    """
    max_similarity = 0
    for ref in reference_list:
        similarity = SequenceMatcher(None, query.lower(), ref.lower()).ratio()
        max_similarity = max(max_similarity, similarity)
        if max_similarity >= threshold:
            return True, max_similarity
    return False, max_similarity

# Check overlap for each generated query
overlapping_queries = []
highest_similarity = 0
highest_sim_query = None
highest_sim_idx = None

for idx, query in enumerate(all_aug_query):
    has_overlap, similarity = check_content_overlap(query, all_query)
    if similarity > highest_similarity:
        highest_similarity = similarity
        highest_sim_query = query
        highest_sim_idx = idx
    if has_overlap:
        overlapping_queries.append((idx, query, similarity))

# Print results
print(f"Found {len(overlapping_queries)} queries with high overlap")
print(f"Total queries checked: {len(all_aug_query)}")
print(f"\nHighest similarity score: {highest_similarity:.3f}")
print(f"Query with highest similarity (idx {highest_sim_idx}):")
print(f"Generated: {highest_sim_query}")

# Print some examples of overlapping content
print("\nOther overlapping examples:")
for idx, query, sim in overlapping_queries[:5]:  # Show first 5 examples
    print(f"\nQuery {idx} (similarity: {sim:.3f}):")
    print(f"Generated: {query}")