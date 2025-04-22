#!/bin/bash

# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Mistral_long.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Dream_long.json
torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/MetaLlama3_long.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Qwen2_long.json

# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/MetaLlama3_if.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Qwen2_if.json




# Define variables
# PEFT_MODEL_PATH="./output/qwen/Qwen2.5-7B-Instruct-mntp-unsup-simcse/checkpoint-1000"
# BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
# PEFT_MODEL="/home/siyue001/Projects/llm2vec_reason_dream/output/simcse/Meta-Llama-3-8B-Instruct-mntp-simcse-TheoremAug-all-v0.1/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-3_s-42_w-50_lr-0.0001_lora_r-16/checkpoint-800"
# BATCH_SIZE=10


# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "/home/siyue/Projects/llm2vec_reason/output/supervised/Meta-Llama-3-8B-Instruct-mntp-supervised-theoremqa-theorems-v1.0/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-10_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-400" \
#     --task_name "BrightTheoremqaTheorems" \
#     --output_dir "results/in_domain/supervised_theoremqa_theorems_v1.0_400" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "/home/siyue/Projects/llm2vec_reason/output/supervised/Meta-Llama-3-8B-Instruct-mntp-supervised-theoremqa-theorems-v1.0/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-10_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000" \
#     --task_name "BrightTheoremqaTheorems" \
#     --output_dir "results/in_domain/supervised_theoremqa_theorems_v1.0_1000" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "output/supervised/Meta-Llama-3-8B-Instruct-mntp-supervised-leetcode-v0.3/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-3_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-200" \
#     --task_name "BrightLeetcode" \
#     --output_dir "results/in_domain/supervised_leetcode_v1.0_200" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "/home/siyue/Projects/llm2vec_reason/output/simcse/Meta-Llama-3-8B-Instruct-mntp-simcse-leetcode-v1.0-hard/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-10_s-42_w-50_lr-0.0001_lora_r-16/checkpoint-210" \
#     --task_name "BrightLeetcode" \
#     --output_dir "results/in_domain/simcse_leetcode_v1.0_hard_210" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "/home/siyue/Projects/llm2vec_reason/output/simcse/Meta-Llama-3-8B-Instruct-mntp-simcse-leetcode-v1.0/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-3_s-42_w-50_lr-0.0001_lora_r-16/checkpoint-741" \
#     --task_name "BrightLeetcode" \
#     --output_dir "results/in_domain/simcse_leetcode_v1.0_741" \
#     --batch_size "$BATCH_SIZE"

# # Run the command
# python experiments/mteb_eval_custom.py \
#     --base_model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct"\
#     --peft_model_name_or_path "$PEFT_MODEL" \
#     --task_name "BrightTheoremqaQuestions" \
#     --output_dir "results/simcse/TheoremAug-all-v0.1_800_theoremqa_questions" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --base_model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct"\
#     --peft_model_name_or_path "$PEFT_MODEL" \
#     --task_name "BrightAops" \
#     --output_dir "results/simcse/TheoremAug-all-v0.1_800_aops" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --base_model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct"\
#     --peft_model_name_or_path "$PEFT_MODEL" \
#     --task_name "BrightLeetcode" \
#     --output_dir "results/simcse/TheoremAug-all-v0.1_800_leetcode" \
#     --batch_size "$BATCH_SIZE"


# python experiments/mteb_eval_custom.py \
#     --fast_bright_root "/home/siyue001/Projects/llm2vec_reason_dream/cache/fast_bright" \
#     --base_model_name_or_path "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"\
#     --peft_model_name_or_path "/home/siyue001/Projects/llm2vec_reason_dream/output/simcse/Meta-Llama-3-8B-Instruct-mntp-simcse-TheoremAug-deepseek-all-v0.1/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-4096_bidirectional-True_e-3_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000" \
#     --task_name "BrightLeetcode" \
#     --output_dir "fast_results/exp/leetcode/Llama-3-8B-Instruct-mntp-simcse-TheoremAug-deepseek-all-v0.1-1000" \
#     --batch_size 16

# python experiments/mteb_eval_custom.py \
#     --fast_bright_root "/home/siyue001/Projects/llm2vec_reason_dream/cache/fast_bright" \
#     --base_model_name_or_path "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"\
#     --peft_model_name_or_path /home/siyue001/Projects/llm2vec_reason_dream/output/simcse/Meta-Llama-3-8B-Instruct-mntp-simcse-TheoremAug-4omini-all-v0.2/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-3000_bidirectional-True_e-3_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-600 \
#     --task_name "BrightTheoremqaQuestions" \
#     --output_dir "fast_results/exp/theoremqa_questions/Llama-3-8B-Instruct-mntp-simcse-TheoremAug-4omini-all-v0.2-600" \
#     --batch_size 16

# python experiments/mteb_eval_custom.py \
#     --base_model_name_or_path "Qwen/Qwen2.5-7B-Instruct"\
#     --peft_model_name_or_path "/home/siyue001/Projects/llm2vec_reason_dream/output/qwen/Qwen2.5-7B-Instruct-mntp-TheoremAug-4omini-all-v0.1/E5Mix_train_m-Qwen2.5-7B-Instruct_p-mean_b-32_l-3000_bidirectional-True_e-3_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-800" \
#     --task_name "BrightTheoremqaTheorems" \
#     --output_dir "results/exp/theoremqa_theorems/Qwen2.5-7B-Instruct-mntp-TheoremAug-4omini-all-v0.1-800" \
#     --batch_size 16

#     --fast_bright_root "/home/siyue001/Projects/llm2vec_reason_dream/cache/fast_bright" \


# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --base_model_name_or_path "$BASE_MODEL"\
#     --task_name "BrightAops" \
#     --output_dir "qwen_results/simcse/aops" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightPony" \
#     --output_dir "results/all_domain/supervised_new_pony_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --base_model_name_or_path "$BASE_MODEL"\
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightLeetcode" \
#     --output_dir "qwen_results/simcse/leetcode" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightStackOverflow" \
#     --output_dir "results/all_domain/supervised_new_stackoverflow_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightEconomics" \
#     --output_dir "results/all_domain/supervised_new_economics_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "$PEFT_MODEL_PATH" \
#     --task_name "BrightBiology" \
#     --output_dir "results/all_domain/supervised_new_biology_2498" \
#     --batch_size "$BATCH_SIZE"

# python experiments/mteb_eval_custom.py \
#     --peft_model_name_or_path "/home/siyue/Projects/llm2vec_reason/output/simcse/Meta-Llama-3-8B-Instruct-mntp-simcse-new-v0.1/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-24_l-4096_bidirectional-True_e-1_s-42_w-300_lr-0.0001_lora_r-16/checkpoint-2498" \
#     --task_name "BrightBiology" \
#     --output_dir "results/all_domain/simcse_new_biology_2498" \
#     --batch_size "$BATCH_SIZE"

    