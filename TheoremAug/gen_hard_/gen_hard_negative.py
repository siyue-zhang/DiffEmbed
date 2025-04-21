import pickle
import random
import json

from collections import defaultdict, Counter
from itertools import permutations, combinations
from copy import deepcopy
import bm25s

import sys
sys.path.append("/home/siyue001/Projects/llm2vec_reason_dream/TheoremAug/")
from instances import *
from util import separate, load_jsonl, write_jsonl


# file_path = '../gen_problem_solution/check_solution_input.jsonl'
# data_path = '../gen_problem_solution/check_problem_solution_tmp.jsonl'
# ids = [d['custom_id'] for d in load_jsonl(file_path)]

# before_problem_by_instance = defaultdict(list)
# for example in load_jsonl(data_path):
#     if str(example['id']) in ids:
#         instance = example['instance']
#         problem = example['problem']
#         if problem not in before_problem_by_instance[instance]:
#             before_problem_by_instance[instance].append(problem)


file_path = "../gen_problem_solution/before_hard_negative_tmp.pkl"

# Load from a file
with open(file_path, "rb") as file:
    data = pickle.load(file)

# Retrieve objects
pairs_by_instance = data["pairs_by_instance"]
problem_by_instance = data["problem_by_instance"]
solution_by_instance = data["solution_by_instance"]
definition_by_instance = data["definition_by_instance"]

get_domain_by_instance={}
for domain in domains:
    list_instances = domains[domain]
    for instance in list_instances:
        get_domain_by_instance[instance]=domain


definition_instances = list(definition_by_instance.keys())
definition_corpus = list(definition_by_instance.values())
definition_retriever = bm25s.BM25()
definition_retriever.index(bm25s.tokenize(definition_corpus))

problem_instances = []
problem_corpus = []
for instance, problems in problem_by_instance.items():
    for problem in problems:
        problem_instances.append(instance)
        problem_corpus.append(problem)
problem_retriever = bm25s.BM25()
problem_retriever.index(bm25s.tokenize(problem_corpus))

solution_instances = []
solution_corpus = []
for instance, solutions in solution_by_instance.items():
    for solution in solutions:
        solution_instances.append(instance)
        solution_corpus.append(solution)
solution_retriever = bm25s.BM25()
solution_retriever.index(bm25s.tokenize(solution_corpus))

skip = 2
examples_by_task = defaultdict(list)


for instance, pairs in pairs_by_instance.items():
    domain = pairs[0]['domain']
    question_type = pairs[0]['question_type']
    reference = pairs[0]['reference']

    # p2i: theoremqa_theorems

    definition = definition_by_instance[instance]
    problems = [pair['problem'] for pair in pairs]
    solutions = [pair['solution'] for pair in pairs]

    if len(problems) != len(list(set(problems))):
        unique_problems = []
        unique_solutions = []
        for problem, solution in zip(problems, solutions):
            if problem not in unique_problems:
                unique_problems.append(problem)
                unique_solutions.append(solution)
        problems = unique_problems
        solutions = unique_solutions

    negatives_by_problem = defaultdict(list)
    negatives_instance_by_problem = defaultdict(list)
    n_neg = 2
    for problem in problems:
        results, scores = definition_retriever.retrieve(bm25s.tokenize(problem), k=200)
        ids = results[0]

        skip_instances = []
        for k, idx in enumerate(ids):
            ins = definition_instances[idx]
            if ins == instance:
                continue
            if ins not in skip_instances:
                skip_instances.append(ins)
            if len(skip_instances)==skip:
                skip_k=k+1
                break

        for k, idx in enumerate(ids):
            if k<skip_k:
                continue
            if definition_instances[idx] == instance:
                continue
            if definition_instances[idx] in negatives_instance_by_problem[problem]:
                continue
            if get_domain_by_instance[definition_instances[idx]] != get_domain_by_instance[instance]:
                continue
            negatives_by_problem[problem].append(definition_corpus[idx])
            negatives_instance_by_problem[problem].append(definition_instances[idx])
            if len(negatives_by_problem[problem])==n_neg:
                break

        # if instance == "Kahn's Algorithm":
        #     print(skip_instances, skip_k)
        #     print(negatives_instance_by_problem[problem])
        #     assert 1==2

    for problem in problems:
        while len(negatives_by_problem[problem])>0:
            n = negatives_by_problem[problem].pop(0)
            i = negatives_instance_by_problem[problem].pop(0)
            d = deepcopy(definition)
            nc = n.split('**Definition**')[0].replace('**Concept**','').strip()
            nf = n.split('**Definition**')[1].strip()
            dc = d.split('**Definition**')[0].replace('**Concept**','').strip()
            df = d.split('**Definition**')[1].strip()
            if random.random() < 0.5:
                d=df
                n=nf
            else:
                d=dc
                n=nc
            examples_by_task['p2i'].append({
                'instance': instance,
                'domain': domain,
                'question_type': question_type,
                'reference': reference,
                'user_query': problem,
                'positive_document': d,
                'hard_negative_document': n,
                'negative_instance': i,
            })
        # print(f'\n\nquery: \n{problem} \npos {flg}: \n{definition if flg else instance} \nneg: \n{n}\n{i}\n~~~~~~~')

    # print('p2i ', len(examples_by_task['p2i']))


    # i2ps: preproc_aops, preproc_leetcode, preproc_theoremqa_questions

    if len(solutions)>0:
        results, scores = solution_retriever.retrieve(bm25s.tokenize(definition), k=1000)
        ids = results[0]

        negatives = []
        negatives_problem = []
        negatives_instance = []
        n_neg = len(solutions)

        skip_instances = []
        for k, idx in enumerate(ids):
            ins = solution_instances[idx]
            if ins == instance:
                continue
            if ins not in skip_instances:
                skip_instances.append(ins)
            if len(skip_instances)==skip:
                skip_k=k+1
                break

        for k, idx in enumerate(ids):
            if k<skip_k:
                continue
            if solution_instances[idx]==instance:
                continue
            if get_domain_by_instance[solution_instances[idx]] != get_domain_by_instance[instance]:
                continue
            if solution_instances[idx] in negatives_instance:
                continue
            negatives.append(solution_corpus[idx])
            negatives_problem.append(problem_corpus[idx])
            negatives_instance.append(solution_instances[idx])
            if len(negatives)==n_neg:
                break
        
        idx = list(range(len(solutions)))
        for neg_solution, neg_instance, neg_problem in zip(negatives, negatives_instance, negatives_problem):
            index = random.choice(range(len(idx)))
            k = idx.pop(index)
            pos_problem = problems[k]
            pos_solution = solutions[k]
            if question_type=='coding':
                pos = pos_solution
                neg = neg_solution
            else:
                pos = pos_problem+'\n'+pos_solution
                neg = neg_problem+'\n'+neg_problem
            res = {
                'instance': instance,
                'domain': domain,
                'question_type': question_type,
                'reference': reference,
                'user_query': instance,
                'positive_document': pos,
                'hard_negative_document': neg,
                'negative_instance': neg_instance,
            }

            examples_by_task['i2ps'].append(res)
            tmp_pos = res['positive_document']
            tmp_neg = res['hard_negative_document']
            q = res['user_query']
            # print(f'\nquery: \n{q} \npos: \n{tmp_pos} \nneg: \n{tmp_neg}\nneg instance:\n{neg_instance}\n')
        
        # print('i2ps ', len(examples_by_task['i2ps']))


    # p(s)(i)2ps: leetcode, aops
    
    negatives_problem_by_problem = defaultdict(list)
    negatives_instance_by_problem = defaultdict(list)
    negatives_solution_by_problem = defaultdict(list)
    n_neg = len(problems)-1
    for problem in problems:
        results, scores = problem_retriever.retrieve(bm25s.tokenize(problem), k=1000)
        ids = results[0]

        skip_instances = []
        for k, idx in enumerate(ids):
            ins = problem_instances[idx]
            if ins == instance:
                continue
            if ins not in skip_instances:
                skip_instances.append(ins)
            if len(skip_instances)==skip:
                skip_k=k+1
                break

        for k, idx in enumerate(ids):
            if k<skip_k:
                continue
            if problem_instances[idx] == instance:
                continue
            if problem_corpus[idx] == problem:
                continue
            if problem_instances[idx] in negatives_instance_by_problem[problem]:
                continue
            if get_domain_by_instance[problem_instances[idx]] != get_domain_by_instance[instance]:
                continue
            negatives_problem_by_problem[problem].append(problem_corpus[idx])
            negatives_instance_by_problem[problem].append(problem_instances[idx])
            negatives_solution_by_problem[problem].append(solution_corpus[idx])
            if len(negatives_problem_by_problem[problem])==n_neg:
                break

    pair_combinations = list(permutations(range(len(problems)), 2))
    pair_combinations = random.sample(pair_combinations, min(len(pair_combinations), 20))
    
    for query_idx, pos_idx in pair_combinations:
        task = 'p2ps' if random.random() < 0.75 else 'ps2ps'
        query_problem = problems[query_idx]
        query_solution = solutions[query_idx]
        pos_problem = problems[pos_idx]
        pos_solution = solutions[pos_idx]

        neg_problem = negatives_problem_by_problem[query_problem].pop(0)
        neg_solution = negatives_solution_by_problem[query_problem].pop(0)
        neg_instance = negatives_instance_by_problem[query_problem].pop(0)

        if question_type=='coding':
            pos = pos_solution
            neg = neg_solution
        else:
            pos = pos_problem+'\n'+pos_solution
            neg = neg_problem+'\n'+neg_solution

        if task == 'p2ps':
            user_query = query_problem
        else:
            if random.random() < 0.6:
                user_query = query_problem+'\n'+query_solution
            else:
                user_query = query_problem+'\n'+instance

        res = {
            'instance': instance,
            'domain': domain,
            'question_type': question_type,
            'reference': reference,
            'user_query': user_query,
            'positive_document': pos,
            'hard_negative_document': neg,
            'negative_instance': neg_instance,
        }
        examples_by_task[task].append(res)
        # p = res['positive_document']
        # n = res['hard_negative_document']
        # i = res['negative_instance']
        # print(f'\n\nquery: \n{query_problem} \n\npos: \n{p} \n\nneg: \n{n}\n{i}\n~~~~~~~')


    pair_combinations = [(i, (i + 1) % len(problems)) for i in range(len(problems))]
    # print(pair_combinations)

    for query_idx, pos_idx in pair_combinations:
        task = 'p2ps' if random.random() < 0.75 else 'ps2ps'
        query_problem = problems[query_idx]
        query_solution = solutions[query_idx]
        pos_problem = problems[pos_idx]
        pos_solution = solutions[pos_idx]

        if question_type=='coding':
            pos = pos_solution
        else:
            pos = pos_problem+'\n'+pos_solution

        if task == 'p2ps':
            user_query = query_problem
        else:
            if random.random() < 0.6:
                user_query = query_problem+'\n'+query_solution
            else:
                user_query = query_problem+'\n'+instance

        res = {
            'instance': instance,
            'domain': domain,
            'question_type': question_type,
            'reference': reference,
            'user_query': user_query,
            'positive_document': pos,
            'hard_negative_document': None,
            'negative_instance': 'GENERATED',
        }
        examples_by_task[task].append(res)

        # # p(s)2ps
        
        # pair_combinations = list(permutations(range(len(problems)), 2))
        # for query_idx, pos_idx in pair_combinations:
        #     task = random.choice(['p2ps', 'ps2ps'])
        #     query_problem = problems[query_idx]
        #     query_solution = solutions[query_idx]
        #     pos_problem = problems[pos_idx]
        #     pos_solution = solutions[pos_idx]
        #     if question_type=='coding':
        #         pos = pos_solution
        #     else:
        #         pos = pos_problem+'\n'+pos_solution
        #     if task == 'p2ps':
        #         user_query = query_problem
        #     else:
        #         user_query = query_problem+'\n'+query_solution
        #     res = {
        #         'instance': instance,
        #         'domain': domain,
        #         'question_type': question_type,
        #         'reference': reference,
        #         'user_query': user_query,
        #         'positive_document': pos,
        #         'hard_negative_document': None,
        #         'negative_instance': None,
        #     }

        #     examples_by_task[task].append(res)


def get_instruct_map(domain, question_type):
    instruct_map = {
        'p2i': f'Gievn a {question_type} problem, retrieve the relevant {domain} that help solve the given problem.',
        'p2ps': f'Given a {question_type} problem, retrieve the relevant problems that can be solved by the similar {domain}.',
        'i2ps': f'Given a {domain}, retrieve the relevant {question_type} problems that apply the given {domain}.',
        'ps2ps': f'Given a problem with a solution, retrieve the relevant {question_type} problems that can be solved by the similar {domain}.',
    }
    return instruct_map



finals=[]
gen_finals = []
requests = []
count_example = 0
for task, examples in examples_by_task.items():
    for example in examples:
        count_example += 1
        example['id'] = str(count_example)
        domain =  example['domain']
        question_type = example['question_type']
        instance = example['instance']
        query = example['user_query']
        pos = example['positive_document']
        neg = example['hard_negative_document']
        instruction = get_instruct_map(domain, question_type)[task]

        final = {
            'id': example['id'],
            'domain': example['domain'],
            'instance': example['instance'],
            'task_type': task,
            'task': instruction,
            'user_query': example['user_query'],
            'positive_document': pos,
            'hard_negative_document':neg,
            'negative_instance': example['negative_instance']
        }
        finals.append(final)


        if neg==None:
            gen_finals.append(final)
            prompt = f"""You have been assigned a retrieval task: {instruction}
You will be given a user query and a positive document. Your mission is to write one hard hard negative document. The hard negative document must:
- Have the similar context background as the user query but test or require a different {domain}.
- Follow the format of the positive document.
- Should not be related to {instance}.
- Should not be helpful for solving the user query problem.

**User Query**
{query}

**Positive Document**
{pos}

Directly response the content of hard negative document.
**Hard Negative Document**
"""
            messages = [
                {"role": "system", "content": "You are an expert for generating hard negative document for information retrieval tasks."},
                {"role": "user", "content": prompt},
                ]
            requests.append({
                "custom_id": example['id'], 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": 
                    {"model": "gpt-4o-mini", 
                        "messages": messages,
                    "max_tokens": 4000},
            })

print('\n\n')     
print(Counter([d['domain'] for d in gen_finals]))
print(Counter([d['task_type'] for d in gen_finals]))
print(len(gen_finals))
print('\n')

print(Counter([d['domain'] for d in finals]))
print(Counter([d['task_type'] for d in finals]))
print(len(finals))
print('\n')

file_path = '../output/TheoremAug_tmp.jsonl'
write_jsonl(file_path, finals)

        
file_path = 'gen_hard_negative_input.jsonl'
write_jsonl(file_path, requests)




# for task, examples in examples_by_task.items():

#     for example in examples:
#         if example['domain']=='physics theorem' and random.random() > 0.2:
#             continue
#         if task =='p2ps' and random.random() > 0.7:
#             continue
#         if task == 'ps2ps' and random.random() > 0.3:
#             continue
#         if example['domain']=='finance formula' and example['instance'] not in F:
#             continue

#         # if example['domain'] not in ['algorithm', 'data structure']:
#         #     continue
#         # if task != 'p2ps':
#         #     continue

#         if task not in ['p2i','i2ps']:
#             continue

#         final = {
#             'domain': example['domain'],
#             'instance': example['instance'],
#             'task_type': task,
#             'task': get_instruct_map(example['domain'], example['question_type'])[task],
#             'user_query': example['user_query'],
#             'positive_document': example['positive_document'],
#             'hard_negative_document': example['hard_negative_document'],
#             'negative_instance': example['negative_instance']
#         }

#         finals.append(final)


