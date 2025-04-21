import pickle
import ipdb
import random
import json

from collections import defaultdict, Counter
from itertools import permutations
from copy import deepcopy

import sys
sys.path.append('../')
from instances import *

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

file_path = '../gen_problem_solution/check_solution_input.jsonl'
data_path = '../gen_problem_solution/check_problem_solution_tmp.jsonl'
ids = [d['custom_id'] for d in load_jsonl(file_path)]

before_problem_by_instance = defaultdict(list)
for example in load_jsonl(data_path):
    if str(example['id']) in ids:
        instance = example['instance']
        problem = example['problem']
        if problem not in before_problem_by_instance[instance]:
            before_problem_by_instance[instance].append(problem)



filepath = "../gen_problem_solution/before_hard_negative_tmp.pkl"

# Load from a file
with open(filepath, "rb") as file:
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

import bm25s

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

skip = 10
examples_by_task = defaultdict(list)


# for instance in pairs_by_instance:
#     if len(pairs_by_instance[instance])>8:
#         pairs_by_instance[instance] = random.sample(pairs_by_instance[instance], 8)

for instance, pairs in pairs_by_instance.items():
    domain = pairs[0]['domain']
    question_type = pairs[0]['question_type']
    reference = pairs[0]['reference']
    language = pairs[0]['language']

    # p2i: theoremqa_theorems

    definition = definition_by_instance[instance]
    problems = [pair['problem'] for pair in pairs]
    solutions = [pair['solution'] for pair in pairs]

    before_problems = before_problem_by_instance[instance]

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
    n_neg = 5
    for problem in before_problems:
        results, scores = definition_retriever.retrieve(bm25s.tokenize(problem), k=200)
        ids = results[0]
        for k, idx in enumerate(ids):
            if k<2:
                continue
            if definition_instances[idx] == instance:
                continue
            if get_domain_by_instance[definition_instances[idx]] != get_domain_by_instance[instance]:
                continue
            negatives_by_problem[problem].append(definition_corpus[idx])
            negatives_instance_by_problem[problem].append(definition_instances[idx])
            if len(negatives_by_problem[problem])==n_neg:
                break
    
    for problem in before_problems:
        while len(negatives_by_problem[problem])>0:
            n = negatives_by_problem[problem].pop(0)
            i = negatives_instance_by_problem[problem].pop(0)
            d = deepcopy(definition)
            if random.random() < 0.8:
                nc = n.split('**Definition**')[0].replace('**Concept**','').strip()
                nf = n.split('**Definition**')[1]
                dc = d.split('**Definition**')[0].replace('**Concept**','').strip()
                df = d.split('**Definition**')[1]
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

    print('p2i ', len(examples_by_task['p2i']))

    # # i2p

    # if len(problems)>0:
    #     results, scores = problem_retriever.retrieve(bm25s.tokenize(definition), k=1000)
    #     ids = results[0]

    #     negatives = []
    #     negatives_instance = []
    #     n_neg = 5*len(problems)
    #     for k, idx in enumerate(ids):
    #         if k<skip:
    #             continue
    #         if problem_instances[idx]==instance:
    #             continue
    #         if problem_instances[idx] in negatives_instance:
    #             continue
    #         if get_domain_by_instance[problem_instances[idx]] != get_domain_by_instance[instance]:
    #             continue
    #         negatives.append(problem_corpus[idx])
    #         negatives_instance.append(problem_instances[idx])
    #         if len(negatives)==n_neg:
    #             break
        
    #     for neg, neg_instance in zip(negatives, negatives_instance):
    #         flg = random.random() < 0.5
    #         res = {
    #             'instance': instance,
    #             'domain': domain,
    #             'question_type': question_type,
    #             'reference': reference,
    #             'user_query': definition if flg else instance,
    #             'positive_document': random.choice(problems),
    #             'hard_negative_document': neg,
    #             'negative_instance': neg_instance,
    #         }
    #         examples_by_task['i2p'].append(res)
    #         tmp_pos = res['positive_document']
    #         tmp_neg = res['hard_negative_document']
    #         # print(f'\n\nquery: \n{definition if flg else instance} \npos: \n{tmp_pos} \nneg: \n{tmp_neg}\n')

    #     print(len(examples_by_task['i2p']))

    # p2ps: leetcode, aops
    
    negatives_problem_by_problem = defaultdict(list)
    negatives_instance_by_problem = defaultdict(list)
    negatives_solution_by_problem = defaultdict(list)
    n_neg = len(problems)-1
    for problem in problems:
        results, scores = problem_retriever.retrieve(bm25s.tokenize(problem), k=1000)
        ids = results[0]
        for k, idx in enumerate(ids):
            if k<skip:
                continue
            if problem_instances[idx] == instance:
                continue
            if problem_corpus[idx] == problem:
                continue
            if random.choice([True, False]):
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
    
    for query_idx, pos_idx in pair_combinations:
        query_problem = problems[query_idx]
        p = problems[pos_idx]
        ps = solutions[pos_idx]
        n = negatives_problem_by_problem[query_problem].pop(0)
        ns = negatives_solution_by_problem[query_problem].pop(0)
        neg_instance = negatives_instance_by_problem[query_problem].pop(0)
        # if domain in ['algorithm', 'data structure']:
        #     pos = ps
        #     neg = ns
        # else:
        pos = p+'\n'+ps
        neg = n+'\n'+ns
        res = {
            'instance': instance,
            'domain': domain,
            'question_type': question_type,
            'reference': reference,
            'user_query': query_problem,
            'positive_document': pos,
            'hard_negative_document': neg,
            'negative_instance': neg_instance,
        }
        examples_by_task['p2ps'].append(res)
        p = res['positive_document']
        n = res['hard_negative_document']
        i = res['negative_instance']
        # print(f'\n\nquery: \n{query_problem} \n\npos: \n{p} \n\nneg: \n{n}\n{i}\n~~~~~~~')

    print('p2ps ',len(examples_by_task['p2ps']))

    # ps2ps

    negatives_problem_by_problem = defaultdict(list)
    negatives_instance_by_problem = defaultdict(list)
    negatives_solution_by_problem = defaultdict(list)
    n_neg = len(problems)-1
    for problem, solution in zip(problems, solutions):
        results, scores = solution_retriever.retrieve(bm25s.tokenize(solution), k=1000)
        ids = results[0]
        for k, idx in enumerate(ids):
            if k<skip:
                continue
            if solution_instances[idx] == instance:
                continue
            if solution_corpus[idx] == problem:
                continue
            if random.choice([True, False]):
                if solution_instances[idx] in negatives_instance_by_problem[problem]:
                    continue
            if get_domain_by_instance[solution_instances[idx]] != get_domain_by_instance[instance]:
                continue
            negatives_problem_by_problem[problem].append(problem_corpus[idx])
            negatives_instance_by_problem[problem].append(problem_instances[idx])
            negatives_solution_by_problem[problem].append(solution_corpus[idx])
            if len(negatives_problem_by_problem[problem])==n_neg:
                break
        

    pair_combinations = list(permutations(range(len(problems)), 2))

    for query_idx, pos_idx in pair_combinations:
        query_problem = problems[query_idx]
        query_solution = solutions[query_idx]
        p = problems[pos_idx]
        ps = solutions[pos_idx]
        n = negatives_problem_by_problem[query_problem].pop(0)
        ns = negatives_solution_by_problem[query_problem].pop(0)
        neg_instance = negatives_instance_by_problem[query_problem].pop(0)
        # if domain in ['algorithm', 'data structure']:
        #     pos = ps
        #     neg = ns
        # else:
        pos = p+'\n'+ps
        neg = n+'\n'+ns
        res = {
            'instance': instance,
            'domain': domain,
            'question_type': question_type,
            'reference': reference,
            'user_query': query_problem+'\n'+query_solution,
            'positive_document': pos,
            'hard_negative_document': neg,
            'negative_instance': neg_instance,
        }
        examples_by_task['ps2ps'].append(res)
        # p = res['positive_document']
        # n = res['hard_negative_document']
        # i = res['negative_instance']
        # print(f'\n\nquery: \n{query_problem} \npos \n{p} \nneg: \n{n}\n{i}\n~~~~~~~')

    print('ps2ps', len(examples_by_task['ps2ps']))

    # # p2s

    # problem_negative_memory = defaultdict(list)
    # for pair in pairs:
    #     problem = pair['problem']
    #     solution = pair['solution']
    #     # other_instance = random.choice([True, False])
    #     other_instance = True
    #     results, scores = solution_retriever.retrieve(bm25s.tokenize(problem), k=1000)
    #     ids = results[0]
    #     for k, idx in enumerate(ids):
    #         if k<skip:
    #             continue
    #         if solution_corpus[idx] == solution:
    #             continue
    #         # can be the same algorithm but wrong solution
    #         if other_instance and solution_instances[idx] == instance:
    #             continue
    #         if solution_corpus[idx] in problem_negative_memory[problem]:
    #             continue
    #         if get_domain_by_instance[problem_instances[idx]] != get_domain_by_instance[instance]:
    #             continue
    #         break
        
    #     problem_negative_memory[problem].append(solution_corpus[idx])
    #     res = {
    #         'instance': instance,
    #         'domain': domain,
    #         'question_type': question_type,
    #         'reference': reference,
    #         'user_query': problem,
    #         'positive_document': solution,
    #         'hard_negative_document': solution_corpus[idx],
    #         'negative_instance': solution_instances[idx]
    #     }
    #     examples_by_task['p2s'].append(res)
    #     n = res['hard_negative_document']
    #     i = res['negative_instance']
    #     # print(f'\n\nquery: \n{problem} \npos \n{solution} \nneg: \n{n}\n{i}\n~~~~~~~')

    # print(len(examples_by_task['p2s']))

    # p2ps: leetcode, aops

    # negatives_by_problem = defaultdict(list)
    # negatives_problem_by_problem = defaultdict(list)
    # negatives_instance_by_problem = defaultdict(list)
    # n_neg = len(problems)-1
    # for problem, solution in zip(problems, solutions):
    #     results, scores = solution_retriever.retrieve(bm25s.tokenize(problem), k=1000)
    #     # print(results, scores)
    #     ids = results[0]
    #     for k, idx in enumerate(ids):
    #         if k<skip:
    #             continue
    #         if solution_instances[idx] == instance:
    #             # print('same instance')
    #             continue
    #         if solution_corpus[idx] == solution:
    #             # print('self solution')
    #             continue
    #         if random.choice([True, False]):
    #             if solution_instances[idx] in negatives_instance_by_problem[problem]:
    #                 continue
    #         if get_domain_by_instance[solution_instances[idx]] != get_domain_by_instance[instance]:
    #             continue
    #         negatives_problem_by_problem[problem].append(problem_corpus[idx])
    #         negatives_by_problem[problem].append(solution_corpus[idx])
    #         negatives_instance_by_problem[problem].append(solution_instances[idx])
    #         if len(negatives_by_problem[problem])==n_neg:
    #             break

    # pair_combinations = list(permutations(range(len(problems)), 2))
    # for query_idx, pos_idx in pair_combinations:
    #     query_problem = problems[query_idx]
    #     pos = solutions[pos_idx]
    #     flg = False # random.random() < 0.5
    #     n = negatives_by_problem[query_problem].pop(0)
    #     q = negatives_problem_by_problem[query_problem].pop(0)
    #     res = {
    #         'instance': instance,
    #         'domain': domain,
    #         'question_type': question_type,
    #         'reference': reference,
    #         'user_query': query_problem,
    #         'positive_document': pos if flg else problems[pos_idx]+' '+pos,
    #         'hard_negative_document': n if flg else q+'\n\n'+n,
    #         'negative_instance': negatives_instance_by_problem[query_problem].pop(0)
    #     }
    #     examples_by_task['p2ps'].append(res)
    #     pos = res['positive_document']
    #     n = res['hard_negative_document']
    #     i = res['negative_instance']
    #     # print(f'\n\nquery: \n{query_problem} \npos {instance} \n{pos} \nneg: \n{n}\n{i}\n~~~~~~~')
        
    # print(len(examples_by_task['p2ps']))


    # i2ps: preproc_aops, preproc_leetcode, preproc_theoremqa_questions

    if len(solutions)>0:
        results, scores = solution_retriever.retrieve(bm25s.tokenize(definition), k=1000)
        ids = results[0]

        negatives = []
        negatives_problem = []
        negatives_instance = []
        n_neg = len(solutions)
        for k, idx in enumerate(ids):
            if k<skip:
                continue
            if solution_instances[idx]==instance:
                continue
            # if solution_instances[idx] in negatives_instance:
            #     continue
            if get_domain_by_instance[solution_instances[idx]] != get_domain_by_instance[instance]:
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
            # if domain in ['algorithm', 'data structure']:
            #     pos = pos_solution
            #     neg = neg_solution
            # else:
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

        print('i2ps ', len(examples_by_task['i2ps']))

    # s2s
        
    # negatives_by_solution = defaultdict(list)
    # negatives_instance_by_solution = defaultdict(list)
    # n_neg = len(solutions)-1
    # for solution in solutions:
    #     results, scores = solution_retriever.retrieve(bm25s.tokenize(solution), k=100)
    #     ids = results[0]
    #     for k, idx in enumerate(ids):
    #         if k<skip:
    #             continue
    #         if solution_instances[idx] == instance:
    #             continue
    #         if solution_corpus[idx] == solution:
    #             continue
    #         if solution_instances[idx] in negatives_instance_by_solution[solution]:
    #             continue
    #         if get_domain_by_instance[problem_instances[idx]] != get_domain_by_instance[instance]:
    #             continue
    #         negatives_by_solution[solution].append(solution_corpus[idx])
    #         negatives_instance_by_solution[solution].append(solution_instances[idx])
    #         if len(negatives_by_solution[solution])==n_neg:
    #             break
    
    # pair_combinations = list(permutations(solutions, 2))
    
    # for query_solution, pos_solution in pair_combinations:
    #     res = {
    #         'instance': instance,
    #         'domain': domain,
    #         'question_type': question_type,
    #         'reference': reference,
    #         'user_query': query_solution,
    #         'positive_document': pos_solution,
    #         'hard_negative_document': negatives_by_solution[query_solution].pop(0),
    #         'negative_instance': negatives_instance_by_solution[query_solution].pop(0)
    #     }
    #     examples_by_task['s2s'].append(res)
    #     pos = res['positive_document']
    #     n = res['hard_negative_document']
    #     i = res['negative_instance']
    #     q = res['user_query']
    #     # print(f'\n\nquery: \n{q} \npos {instance} \n{pos} \nneg: \n{n}\n{i}\n~~~~~~~')

    # print(len(examples_by_task['s2s']))


finals=[]
def get_instruct_map(domain):
    instruct_map = {
        'p2i': f'Gievn a problem, retrieve the relevant {domain} that help solve the given problem.',
        'p2ps': f'Given a problem, retrieve the relevant problems that can be solved by the similar {domain}.',
        'i2ps': f'Given a {domain}, retrieve the relevant problems that apply the given {domain}.',
        'ps2ps': f'Given a problem, retrieve the relevant problems that can be solved by the similar {domain}.',

        # 'p2p':f'Given a problem, retrieve the relevant problems that can be solved by the similar {domain}.',
        # 'p2s':f'Given a problem, retrieve the correct solutions to the given problem.',
        # 'p2ps':f'Given a problem, retrieve the solutions to other problems that base on the similar {domain} as the solution to the given question.',
        # 's2s':f'Given a solution, retrieve the solutions that apply the similar {domain} as the given solution.',
        # 'i2p':f'Given a {domain}, retrieve the questions that require the given {domain}.',
        # 'i2s':f'Given a {domain}, retrieve the solutions that apply the given {domain}.',
    }
    return instruct_map

F = ["Binomial Model in finance",
    "Future Value (FV) Formula",
    "Present Value (PV) Formula",
    "Discounted Cash Flow (DCF) Model",
    "Z-Spread Formula",
    "Sharpe Ratio",
    "Payback Period Formula"]


for task, examples in examples_by_task.items():

    for example in examples:
        # if task not in ['p2i','p2ps']:
        #     continue
        # if example['domain'] in ['data structure','algorithm','finance formula']:
        #     continue
        if example['domain']=='physics theorem' and random.random() > 0.2:
            continue
        if task =='p2ps' and random.random() > 0.7:
            continue
        if task == 'ps2ps' and random.random() > 0.3:
            continue
        if example['domain']=='finance formula' and example['instance'] not in F:
            continue

        # if example['domain'] not in ['algorithm', 'data structure']:
        #     continue
        # if task != 'p2ps':
        #     continue

        if task not in ['p2i','i2ps']:
            continue

        final = {
            'domain': example['domain'],
            'instance': example['instance'],
            'task_type': task,
            'task': get_instruct_map(example['domain'])[task],
            'user_query': example['user_query'],
            'positive_document': example['positive_document'],
            'hard_negative_document': example['hard_negative_document'],
            'negative_instance': example['negative_instance']
        }

        finals.append(final)



print(Counter([d['domain'] for d in finals]))
print(Counter([d['task_type'] for d in finals]))

print(len(finals))


import json
file_path = '../output/augmentation_data_p2i.jsonl'
with open(file_path, 'w', encoding='utf-8') as f:
    for request in finals:
        f.write(json.dumps(request) + '\n')
print(f"File saved to {file_path}")


