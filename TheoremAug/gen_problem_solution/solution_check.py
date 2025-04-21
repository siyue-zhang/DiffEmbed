from collections import defaultdict
import sys
sys.path.append("/home/siyue001/Projects/llm2vec_reason_dream/TheoremAug/")
from instances import *
from util import separate, load_jsonl, write_jsonl

data_path = 'problem_check_tmp.jsonl'
problem_path = 'problem_check_output.jsonl'

data = load_jsonl(data_path)
problem_check = load_jsonl(problem_path)

ids = [d['custom_id'] for d in problem_check]
data = [d for d in data if str(d['id']) in ids]

requests = []
all_pairs =[]
update_count = 0
counter = defaultdict(int)
for example, problem_check in zip(data, problem_check):
    assert example['id'] == problem_check['custom_id']
    response = problem_check['response']['body']['choices'][0]['message']['content']
    correct_flg = response.lower().strip().startswith("yes")

    problem = example['problem']
    solution = example['solution']
    domain = example['domain']
    instance = example['instance']
    question_type = example['question_type']

    if not correct_flg:
        # update problem & solution
        update_count += 1
        if response.lower().startswith("no"):
            response = response[2:].strip()
        elif response.lower().startswith("no."):
            response = response[3:].strip()
            
        try:
            problem, solution = separate(response)
        except Exception as e:
            problem, solution = None, None

    if problem!=None and solution!=None:
        example['problem'] = problem
        example['solution'] = solution
        # print('-----')
        # print(solution)
        # input()
        counter[domain] += 1
        if question_type=='coding':
            prompt = f'**Problem**\n{problem}\n**Solution**\n{solution}\n\nIs this a correct solution to the problem and using the {domain} {instance}? If yes, response with the the updated Python solution code, including a docstring that contains the problem description, as well as the input and output specifications. If no, response "NO".'
        else:
            prompt = f'**Problem**\n{problem}\n**Solution**\n{solution}\n\nIs this a correct solution to the problem and using the {domain} {instance}? Response "YES" or "No".'
        messages = [
            {"role": "system", "content": "You are an expert in solving problems."},
            {"role": "user", "content": prompt}
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

        all_pairs.append(example)
    else:
        idx = example['id']
        print(f'sample {idx} instance {instance} fails to separate problem and solution.')

print(f'updated {update_count} problem & solution ...')

print(counter)
print(sum(list(counter.values())))

file_path = 'solution_check_input.jsonl'
write_jsonl(file_path, requests)

data_path = 'solution_check_tmp.jsonl'
write_jsonl(data_path, all_pairs)
