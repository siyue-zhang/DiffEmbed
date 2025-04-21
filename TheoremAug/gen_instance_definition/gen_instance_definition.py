import random
import os
import json
import pickle
import argparse
import time

import sys
sys.path.append('../')
from instances import *
from openai import OpenAI

parser = argparse.ArgumentParser(description='augmentation data generation')
args = parser.parse_args()

def generate_instance_definition_prompt(domain, instance):

    if domain in ['algorithm', 'data structure']:
        prompt = f'Your task is to provide a definition for the {domain}: {instance} with the implementation code.\n\n'
        prompt += """Here are some examples:
**Concept**
Binary Search
**Definition**
Binary Search is an efficient algorithm for finding a target value within a sorted array or list. It works by repeatedly dividing the search interval in half. If the value of the target is less than the middle element of the list, the algorithm continues the search on the left half, otherwise, it continues on the right half. This process is repeated until the target value is found or the interval is empty, which indicates that the target value is not present in the array.\n**Implementation Code**\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n\n    while left <= right:\n        mid = left + (right - left) // 2\n        \n        # Check if the target is present at mid\n        if arr[mid] == target:\n            return mid  # Target found at index mid\n        # If target is greater, ignore the left half\n        elif arr[mid] < target:\n            left = mid + 1\n        # If target is smaller, ignore the right half\n        else:\n            right = mid - 1\n\n    # Target not found\n    return -1\n```

**Concept**
Hash Set
**Definition**
A Hash Set is a data structure that implements a collection of unique elements, designed to provide efficient operations for adding, removing, and checking for the existence of items. It uses a hash table to store the elements, which allows for average time complexity of O(1) for these operations. Each element is hashed to compute an index in an underlying array, where the element is stored. In the event of hash collisions, the structure typically employs techniques such as chaining or open addressing.\n**Implementation Code**\n```python\nclass HashSet:\n    def __init__(self):\n        self.size = 1000  # Initial size of the hash table\n        self.table = [[] for _ in range(self.size)]  # Initialize hash table with empty lists\n    \n    def _hash(self, key):\n        return key % self.size  # Simple hash function\n\n    def add(self, key):\n        index = self._hash(key)\n        if key not in self.table[index]:  # Check for duplicates\n            self.table[index].append(key)  # Append key to the list at the hashed index\n\n    def remove(self, key):\n        index = self._hash(key)\n        if key in self.table[index]:\n            self.table[index].remove(key)  # Remove the key if it exists\n\n    def contains(self, key):\n        index = self._hash(key)\n        return key in self.table[index]  # Return True if the key is found, otherwise False\n```

Here is your task:
**Concept**
"""
        prompt += f'{instance}\n**Definition**\n'
    else:
        prompt = f'Your task is to provide a definition for the {domain}: {instance}. Write the equation in LaTex format.\n\n'
        prompt += """Here are some examples:
**Concept**
Pigeonhole Principle
**Definition**
Let $S$ be a finite set whose cardinality is $n$. Let $S_1, S_2, \ldots, S_k$ be a partition of $S$ into $k$ subsets. Then: :at least one subset $S_i$ of $S$ contains at least $\ceiling {\dfrac n k}$ elements where $\ceiling {\, \cdot \,}$ denotes the ceiling function.

**Concept**
Newton's Second Law of Motion
**Definition**
Newton's Second Law states that the force acting on an object is equal to the mass of the object multiplied by its acceleration. This law explains how the motion of an object changes when subjected to an external force. F = ma, \quad \\text{where } F \\text{ is force (N), } m \\text{ is mass (kg), and } a \\text{ is acceleration (m/s}^2\\text{)}.

Here is your task:
**Concept**
"""
        prompt += f'{instance}\n**Definition**\n'
    return prompt


def generate_requests():
    requests=[]
    counter=0
    for domain in domains:
        for instance in domains[domain]:
            counter += 1
            prompt = generate_instance_definition_prompt(domain, instance)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            requests.append({
                "custom_id": f"request-{counter}", 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": 
                    {"model": "gpt-4o-mini", 
                     "messages": messages,
                    "max_tokens": 1000},
            })
    return requests

requests =  generate_requests()

# Write requests to a JSONL file
file_path = 'instance_definition_input.jsonl'
with open(file_path, 'w', encoding='utf-8') as f:
    for request in requests:
        f.write(json.dumps(request) + '\n')

print(f"File saved to {file_path}")
