import re
import pickle
import argparse
from collections import defaultdict, Counter
import numpy as np
import sys
sys.path.append("/home/siyue001/Projects/llm2vec_reason_dream/TheoremAug/")
from instances import *
from util import separate, load_jsonl, write_jsonl

parser = argparse.ArgumentParser(description='augmentation data generation')
args = parser.parse_args()


tmp_file = "./solution_check_tmp.jsonl"
check_file = "./solution_check_output.jsonl"
definition_file = "../gen_instance_definition/instance_definition_output.jsonl"

tmp_data = load_jsonl(tmp_file)
check_data = load_jsonl(check_file)
def_data = load_jsonl(definition_file)


def extract_code(text):
    pattern = re.compile(r'```(python|java|cpp)?\n(.*?)```', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return text, None
    
    return matches[0][1], matches[0][0]


all_pairs = []
for example, check_solution in zip(tmp_data,check_data):
    # print('---------------')
    assert example['id'] == check_solution['custom_id']
    check = check_solution['response']['body']['choices'][0]['message']['content']
    check = check.strip()
    question_type = example['question_type']

    if question_type=='coding':
        solution, lang = extract_code(check)
        if lang == 'python':
            example['solution'] = solution
        else:
            example['solution'] = None
            example['problem'] = None
    else:
        if check[:3].lower() != 'yes' and '**yes**' not in check.lower():
            example['solution'] = None
            example['problem'] = None            

    if example['problem'] != None and example['solution'] != None:
        if example['problem'][0]=='.':
            example['problem'] = example['problem'][1:].strip()
        all_pairs.append(example)
        # print(example['instance'],'\n~~~~~')
        # print(example['problem'],'\n~~~~~')
        # print(example['solution'])
        # input()

definition_by_instance = {}
for domain in domains:
    list_instances = domains[domain]
    question_type = domain_question_mapping[domain]
    reference = example_datasets[domain]
    for instance in list_instances:
        item = def_data.pop(0)
        response = item['response']['body']['choices'][0]['message']['content']
        definition_by_instance[instance] = response

# check how definition looks like

print(f'{len(tmp_data)} total --> {len(all_pairs)} correct')

n_counter = defaultdict(list)
for pair in all_pairs:
    domain = pair['domain']
    instance = pair['instance']
    n_counter[domain].append(instance)
print(Counter([pair['domain'] for pair in all_pairs]),'\n\n')

for domain in n_counter:
    count = Counter(n_counter[domain])
    average = np.round(sum(count.values()) / len(count),3)
    print(domain,' ',average)
print('\n')

def find_math(all_pairs,ty):
    instances=[pair['instance'] for pair in all_pairs if pair['domain']==ty]
    instances=list(set(instances))
    return len(instances)
print('Algorithm Coverage: ', find_math(all_pairs,'algorithm'), ' / ', find_math(tmp_data,'algorithm'))
print('Math Coverage: ', find_math(all_pairs,'math theorem'), ' / ', find_math(tmp_data,'math theorem'))
print('Physics Coverage: ', find_math(all_pairs,'physics theorem'), ' / ', find_math(tmp_data,'physics theorem'))
print('\n')


from collections import defaultdict
pairs_by_instance = defaultdict(list)
problem_by_instance = defaultdict(list)
solution_by_instance = defaultdict(list)
for pair in all_pairs:
    instance = pair['instance']
    problem = pair['problem']
    domain = pair['domain']
    solution = pair['solution']
    pair['definition'] = definition_by_instance[instance]
    pairs_by_instance[instance].append(pair)
    problem_by_instance[instance].append(problem)
    solution_by_instance[instance].append(solution)

import pickle
with open("before_hard_negative_tmp.pkl", "wb") as file:
    pickle.dump({
        "pairs_by_instance": pairs_by_instance,
        "problem_by_instance": problem_by_instance,
        "solution_by_instance": solution_by_instance,
        "definition_by_instance": definition_by_instance
    }, file)

