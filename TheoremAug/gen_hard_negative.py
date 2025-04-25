import pickle
import random
import json

from collections import defaultdict, Counter
from itertools import permutations, combinations
from copy import deepcopy
import bm25s

from instances import *
from util import get_related_strings, write_jsonl



file_path = "/home/siyue/Projects/diffusion_embedder/TheoremAug/gen_problem_solution/before_hard_negative_tmp.pkl"

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

examples_by_task = defaultdict(list)

p2i_negatives_instance_map = defaultdict(list)

for instance, pairs in pairs_by_instance.items():
    domain = pairs[0]['domain']
    question_type = pairs[0]['question_type']
    reference = pairs[0]['reference']

    # related_instances = get_related_strings(similar_question_mapping, instance)

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
        for k, idx in enumerate(ids):
            if definition_instances[idx] == instance:
                continue
            # if definition_instances[idx] in related_instances:
            #     continue
            if definition_instances[idx] in p2i_negatives_instance_map[instance]:
                continue
            if get_domain_by_instance[definition_instances[idx]] != get_domain_by_instance[instance]:
                continue
            negatives_by_problem[problem].append(definition_corpus[idx])
            negatives_instance_by_problem[problem].append(definition_instances[idx])
            p2i_negatives_instance_map[instance].append(definition_instances[idx])
            if len(negatives_by_problem[problem])==n_neg:
                break

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


    # i2ps

    if len(solutions)>0:
        results, scores = solution_retriever.retrieve(bm25s.tokenize(definition), k=1000)
        ids = results[0]

        negatives = []
        negatives_problem = []
        negatives_instance = []
        n_neg = len(solutions)

        for k, idx in enumerate(ids):
            if solution_instances[idx]==instance:
                continue
            # if solution_instances[idx] in related_instances:
            #     continue
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


    # p(s)(i)2ps: leetcode, aops
    
    negatives_problem_by_problem = defaultdict(list)
    negatives_instance_by_problem = defaultdict(list)
    negatives_solution_by_problem = defaultdict(list)
    n_neg = len(problems)-1
    used_negative_problems = []
    for problem in problems:
        results, scores = problem_retriever.retrieve(bm25s.tokenize(problem), k=1000)
        ids = results[0]

        for k, idx in enumerate(ids):
            if problem_instances[idx] == instance:
                continue
            # if problem_instances[idx] in related_instances:
            #     continue
            if problem_instances[idx] in negatives_instance_by_problem[problem]:
                continue
            if get_domain_by_instance[problem_instances[idx]] != get_domain_by_instance[instance]:
                continue
            if problem_corpus[idx] in used_negative_problems:
                continue
            negatives_problem_by_problem[problem].append(problem_corpus[idx])
            negatives_instance_by_problem[problem].append(problem_instances[idx])
            negatives_solution_by_problem[problem].append(solution_corpus[idx])
            used_negative_problems.append(problem_corpus[idx])
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
    


def get_instruct_map(domain, question_type):
    instruct_map = {
        'p2i': f'Gievn a {question_type} problem, retrieve the relevant {domain} that help solve the given problem.',
        'p2ps': f'Given a {question_type} problem, retrieve the relevant problems that can be solved by the similar {domain}.',
        'i2ps': f'Given a {domain}, retrieve the relevant {question_type} problems that apply the given {domain}.',
        'ps2ps': f'Given a problem with a solution, retrieve the relevant {question_type} problems that can be solved by the similar {domain}.',
    }
    return instruct_map


finals=[]
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


print(Counter([d['domain'] for d in finals]))
print(Counter([d['task_type'] for d in finals]))
print(len(finals))
print('\n')

file_path = 'TheoremAug_test.jsonl'
write_jsonl(file_path, finals)

