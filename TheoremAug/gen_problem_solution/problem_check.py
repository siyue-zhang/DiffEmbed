import glob
from openai import OpenAI
client = OpenAI()

import sys
sys.path.append("/home/siyue001/Projects/llm2vec_reason_dream/TheoremAug/")
from instances import *
from util import separate, load_jsonl, write_jsonl


# gather all inputs
file_paths = glob.glob("inputs/problem_solution_input_*.jsonl")
file_paths = sorted(file_paths)
input_data = {path: load_jsonl(path) for path in file_paths}


# gather all outputs
file_paths = glob.glob("outputs/problem_solution_output_*.jsonl")
file_paths = sorted(file_paths)
output_data = {path: load_jsonl(path) for path in file_paths}


theorem_aug = {}
counter=0
for domain in domains:
    list_instances = domains[domain]
    question_type = domain_question_mapping[domain]
    ref = example_datasets[domain]
    for instance in list_instances:
        counter += 1
        custom_id = f"request-{counter}" 
        theorem_aug[custom_id] = {
            "instance": instance,
            "domain": domain,
            "instance": instance,
            "question_type": question_type,
            "reference": ref,
            "messages": {},
            "pairs":[],
        }

for path in input_data:
    for item in input_data[path]:
        custom_id = item['custom_id']
        idx = path.split('_')[-1]
        theorem_aug[custom_id]['messages'][idx] = item['body']['messages']



for path in output_data:
    for k, item in enumerate(output_data[path]):
        custom_id = item['custom_id']
        input_path = path.replace('output', 'input')
        history = input_data[input_path][k]['body']['messages']
        content = item['response']['body']['choices'][0]['message']['content']
        chat = {"role": "assistant", "content": content}
        try:
            problem, solution = separate(content)
            history.append(chat)
        except Exception as e:
            problem, solution = None, None
        theorem_aug[custom_id]['pairs'].append([problem, solution, history])


all_pairs = []
requests = []
idx = 0
m = 0
# regenerate the missing problems and solutions
for custom_id in theorem_aug:
    pairs = theorem_aug[custom_id]['pairs']
    num = len(pairs)
    for i in range(num):
        if pairs[i][0]==None:
            print(f'sending request...{m}')
            m+=1
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=pairs[i][2],
                max_tokens=4000,
            )
            response = completion.choices[0].message.content
            history = pairs[i][2]
            try:
                problem, solution = separate(response)
                history.append({"role": "assistant", "content": response})
                print('regenerated one problem and solution pair.')
            except Exception as e:
                problem, solution = None, None

            if problem!=None:
                pairs[i] = [problem, solution, history]
    
    for problem, solution, history in pairs:
        if problem!=None and solution!=None:
            idx += 1
            # add the problem and solution to the all_pairs list
            domain = theorem_aug[custom_id]['domain']
            instance = theorem_aug[custom_id]['instance']
            question_type = theorem_aug[custom_id]['question_type']
            reference = theorem_aug[custom_id]['reference']

            # check if the problem is testing the instance
            prompt = f'Is this problem testing or requiring {domain} {instance}? If yes, please answer "YES". If no, please response with a new problem and solution about {instance} with similar context and difficulty. Do not provide any explanation.'
            history.append({"role": "user", "content": prompt})

            requests.append({
                "custom_id": str(idx),
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": 
                    {"model": "gpt-4o-mini", 
                    "messages": history,
                    "max_tokens": 4000},
            })

            item = {
                "id":str(idx),
                "domain": domain,
                "instance": instance,
                "question_type": question_type,
                "reference": reference,
                "problem": problem,
                "solution": solution,
                "history": history,
            }
            all_pairs.append(item)

print(len(all_pairs))

file_path = 'problem_check_tmp.jsonl'
write_jsonl(file_path, all_pairs)

file_path = 'problem_check_input.jsonl'
write_jsonl(file_path, requests)


# def split_problem_solution(data):
#     problems = []
#     solutions = []
#     for item in data:
#         content = item['response']['body']['choices'][0]['message']['content']
#         if ("**Question**:" not in content) or ("**Solution**:" not in content):
#             problems.append(None)
#             solutions.append(None)
#         else:
#             try:
#                 problem, solution = separate(content)
#             except Exception as e:
#                 problem, solution = None, None
#             problems.append(problem)
#             solutions.append(solution)
#     return problems, solutions

# data = [split_problem_solution(d) for d in data]

# meta = []
# for domain in domains:
#     list_instances = domains[domain]
#     question_type = domain_question_mapping[domain]
#     reference = example_datasets[domain]
#     for instance in list_instances:
#         meta.append({
#             "domain": domain,
#             "instance": instance,
#             "question_type": question_type,
#             "reference": reference,
#         })

# all_pairs = []
# id_ = 0
# for pairs in data:
#     metas = deepcopy(meta)
#     tmp = [pairs[0], pairs[1], metas]
#     for problem, solution, meta_ in zip(*tmp):
#         if problem!=None and solution!=None:
#             meta_['problem'] = problem
#             meta_['solution'] = solution
#             id_ += 1
#             meta_['id'] = id_
#             all_pairs.append(meta_)




# requests=[]
# problems_by_instance=defaultdict(list)
# counter = defaultdict(int)
# for example in all_pairs:
#     custom_id = example['id']
#     problem = example['problem']
#     solution = example['solution']
#     domain = example['domain']
#     instance = example['instance']
#     problems_by_instance[instance].append(problem)
#     question_type = example['question_type']
#     if question_type=='finance' and random.random() < 0.25:
#         continue
#     counter[domain] += 1
#     problem_prompt = f'**Problem**\n{problem}\n\nIs this problem testing {domain} {instance}, yes or no?'
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": problem_prompt}
#     ]
#     requests.append({
#         "custom_id": str(custom_id),
#         "method": "POST", 
#         "url": "/v1/chat/completions", 
#         "body": 
#             {"model": "gpt-4o-mini", 
#             "messages": messages,
#             "max_tokens": 100},
#     })

    

# print(counter)
# print(sum(list(counter.values())))






