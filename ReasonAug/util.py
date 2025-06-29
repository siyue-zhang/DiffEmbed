import json
import time
from datetime import datetime

def separate(example):
    problem, solution = None, None
    for tag in ["**Solution:**","**solution:**","**Solution**","**solution**",]:
        if tag in example:
            problem, solution = example.split(tag)
            break
    for tag in ["**problem:**","**Problem:**","**Problem**","**problem**",]:
        if tag in problem:
            problem = problem.replace(tag,'')
            break
    return problem.strip(), solution.strip()

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for request in data:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')
    print(f"File saved to {file_path}")

def wait_until_kam(k):
    print("Starting processing at:", datetime.now())
    """Wait until k AM before proceeding"""
    current_time = datetime.now()
    target_time = current_time.replace(hour=k, minute=0, second=0, microsecond=0)
    
    # If current time is past 1 AM, wait for next day
    if current_time.hour >= k:
        target_time = target_time.replace(day=target_time.day + 1)
    
    # Calculate seconds to wait
    wait_seconds = (target_time - current_time).total_seconds()
    if wait_seconds > 0:
        print(f"Waiting until 1 AM... ({wait_seconds/3600:.2f} hours)")
        time.sleep(wait_seconds)

def get_related_strings(similar_question_mapping, target):
    """
    Returns all related strings for a target that appears as either a key or value
    in similar_question_mapping. The relationship is bidirectional.
    
    Args:
        target (str): The string to find related items for
        
    Returns:
        list[str]: List of all related strings (both keys and values)
    """
    related = set()
    
    # Case 1: target is a key - add all its values
    if target in similar_question_mapping:
        related.update(similar_question_mapping[target])
    
    # Case 2: target is a value - find all keys that contain it 
    for key, values in similar_question_mapping.items():
        if target in values:
            related.add(key)
            # Also add other values associated with this key
            related.update(values)
    
    # Remove the target itself from the results
    if target in related:
        related.remove(target)
        
    return list(related)