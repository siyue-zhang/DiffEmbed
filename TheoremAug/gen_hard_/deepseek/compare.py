import sys
sys.path.append("/home/siyue001/Projects/llm2vec_reason_dream/TheoremAug/")
from instances import *
from util import load_jsonl, my_api_key, get_related_strings, write_jsonl, wait_until_kam


deepseek = load_jsonl('TheoremAug_tmp.jsonl')
openai = load_jsonl('TheoremAug_tmp_test.jsonl')

instance = "Pick's Theorem"

# for ex in openai:
#     if ex['instance']==instance:
#         if ex['task_type']=='p2ps':
#             print(ex)
#             assert 1==2

deepseek = [ex for ex in deepseek if ex['instance']==instance and ex['task_type']=='p2i']
openai = [ex for ex in openai if ex['instance']==instance and ex['task_type']=='p2i']


deepseek = list(set([ex['user_query'] for ex in deepseek]))
openai = list(set([ex['user_query'] for ex in openai]))

for i in range(max(len(deepseek),len(openai))):
    print('------------')
    if i<len(deepseek):
        print('deepseek:')
        print(deepseek[i])

    if i<len(openai):
        print('\nopenai:')
        print(openai[i])