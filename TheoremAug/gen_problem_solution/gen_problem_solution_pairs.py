import random
from collections import Counter

import sys
sys.path.append("/home/siyue001/Projects/llm2vec_reason_dream/TheoremAug/")
from instances import *
from util import separate, load_jsonl, write_jsonl


def pick_language():

    def weighted_random_choice(instances, probabilities):
        return random.choices(instances, weights=probabilities, k=1)[0]

    instances = ['Python', 'Java', 'C++']
    probabilities = [1, 0, 0]
    chosen_instance = weighted_random_choice(instances, probabilities)

    return chosen_instance


def generate_problem_solution_prompt(question_type, domain, instance, ref):
    prompt = f"Your task is to create one {question_type} problem with a correct solution.\n"
    if question_type=='coding':
        language = pick_language()
        options = [
            '\nThe problem should be based on real world human activities.',
            '\nThe problem should be based on a theoretical coding context.',
            '\nThe problem should be about a company or a factory.',
            '\nThe problem should be about a game or a puzzle.',
            '\nThe problem should be about designing a system.',
            '\nThe problem should be about a mathematical task needing automation.',
            '\nThe problem should be about traffic or logistics.',
            '\nThe problem should be about a city or a community.',
            '\nThe problem should be about fiance or business.',
            '\nThe problem should be about software or mobile applications.',
            '\nThe problem should be about education or e-learning.',
            '\nThe problem should be about e-commerce or online marketplaces.',
            '\nThe problem should be about agriculture or food production.',
            '\nThe problem should be about health or fitness.',
            '\nThe problem should be about customer service.',
            '\nThe problem should be about environmental sustainability.',
            '',
        ]
        add = random.choice(options)
        prompt += f"The problem should be new and unique, not similar to common existing problems.{add}\n"
        prompt += f"Most importantly, the problem should require or test about the {domain}: {instance}.\nThe problem should not explicitly mentioning {instance}.\n"
        if random.random() > 0.4:
            add = random.choice(ref)
            prompt += f"The problem should be as difficult as {add}.\n"
        prompt += f"The solution code should be written in the programming language {language}.\n"
    elif question_type!='math':
        add = '\nThe problem should be based on real world human activities.' if random.random() < 0.5 else ''
        prompt += f"The problem should be new and unique, not similar to common existing problems.{add}\n"
        prompt += f"Most importantly, the problem should require or test about the {domain}: {instance}.\nThe problem should not explicitly mentioning {instance}.\nThe problem should involve numerical operations.\nThe problem should ask for only one answer.\n"
        if random.random() > 0.4:
            add = random.choice(ref)
            prompt += f"The problem should be as difficult as {add}.\n"
        if random.random() < 0.5:
            add = 'in multiple steps' if random.random() < 0.7 else f'by multiple {domain}s'
            prompt += f"The problem should be solved {add}.\n"
        prompt += f"The solution should include reasoning or calculation steps.\n"
    else:
        options = [
            '\nThe problem should be based on real world human activities but not a proof problem.',
            '\nThe problem should a multi-choice problem but not a proof problem.',
            '\nThe problem should be theoretical and mathematical but not a proof problem.',
        ]
        options.append(options[-1])
        add = random.choice(options)
        prompt += f"The problem should be new and unique, not similar to common existing problems.{add}\nThe problem should involve numerical operations.\n"
        prompt += f"Most importantly, the problem should require or test about the {domain}: {instance}.\nThe problem should not explicitly mentioning {instance}.\nThe problem should ask for only one answer.\n"
        if random.random() > 0.2:
            add = random.choice(ref)
            prompt += f"The problem should be as difficult as {add}.\n"
        if random.random() < 0.5:
            add = 'in multiple steps' if random.random() < 0.7 else f'by multiple {domain}s'
            prompt += f"The problem should be solved {add}.\n"
        if random.random() < 0.5:
            prompt += f"The solution should not explicitly mention {instance}.\n"
        if random.random() < 0.4:
            prompt += 'The problem should be around four sentences long.\n'
        prompt += f"The solution should include reasoning or calculation steps.\n"
    prompt += """
Write the problem after the **Problem** tag and the solution after the **Solution** tag. Do not write any explanation.
"""

    return prompt


def generate_requests():
    requests=[]
    count_domains=[]
    counter=0
    for domain in domains:
        list_instances = domains[domain]
        question_type = domain_question_mapping[domain]
        ref = example_datasets[domain]
        for instance in list_instances:
            counter += 1
            count_domains.append(domain)
            prompt = generate_problem_solution_prompt(question_type, domain, instance, ref)
            messages = [
                {"role": "system", "content": "You are an expert in drafting novel problems and solutions."},
                {"role": "user", "content": prompt}
            ]
            requests.append({
                "custom_id": f"request-{counter}", 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": 
                    {"model": "gpt-4o-mini",
                    "messages": messages,
                    "max_tokens": 4000},
            })
    print(Counter(count_domains))
    return requests


K = 8
for k in range(K):
    requests =  generate_requests()
    file_path = f'./inputs/problem_solution_input_{k+1}.jsonl'
    write_jsonl(file_path, requests)
