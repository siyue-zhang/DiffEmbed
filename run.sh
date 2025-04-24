#!/bin/bash

## train for instruction following retrieval
## set 4 GPUs

# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Mistral_if.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Dream_if.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/MetaLlama3_if.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Qwen2_if.json

## test for instruction following retrieval
## set 1 GPU

# declare -A MODELS
# MODELS["Qwen/Qwen2.5-7B-Instruct"]=""
# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-msmarco-w-instructions/MSMARCO_train_m-Dream_emb_p-mean_b-128_l-304_bidirectional-True_e-1_s-42_w-20_lr-5e-05_lora_r-32/checkpoint-250"
# MODELS["meta-llama/Meta-Llama-3-8B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Meta-Llama-3-8B-Instruct-mntp-msmarco-w-instructions/MSMARCO_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-128_l-304_bidirectional-True_e-1_s-42_w-20_lr-5e-05_lora_r-32/checkpoint-250"
# MODELS["mistralai/Mistral-7B-Instruct-v0.2"]="/home/siyue/Projects/diffusion_embedder/output/Mistral-7B-Instruct-mntp-msmarco-w-instructions/MSMARCO_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-128_l-304_bidirectional-True_e-1_s-42_w-20_lr-5e-05_lora_r-32/checkpoint-250"

# TASKS=("News21InstructionRetrieval" "Core17InstructionRetrieval" "Robust04InstructionRetrieval" )

# for MODEL in "${!MODELS[@]}"; do
#     PEFT="${MODELS[$MODEL]}"
#     MODEL_NAME=$(basename "$MODEL")

#     # Custom output suffix based on model name
#     if [[ "$MODEL_NAME" == "Dream_emb" ]]; then
#         SUFFIX="msmarco-w-instructions"
#     else
#         SUFFIX="mntp-msmarco-w-instructions"
#     fi

#     for TASK in "${TASKS[@]}"; do
#         echo "Running $TASK with $MODEL_NAME..."
#         python experiments/mteb_eval_custom.py \
#             --base_model_name_or_path "$MODEL" \
#             --peft_model_name_or_path "$PEFT" \
#             --task_name "$TASK" \
#             --output_dir "results/FollowIR/${TASK}/${MODEL_NAME}-${SUFFIX}" \
#             --batch_size 128
#     done
# done

# -----------------------------------------------------------------------------------------#

## train for long document retrieval
# set 4 GPUs

# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Dream_long.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Mistral_long.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/MetaLlama3_long.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Qwen2_long.json

## test for long document retrieval
## set 1 GPU

# declare -A MODELS
# MODELS["Qwen/Qwen2.5-7B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Qwen2.5-7B-Instruct-mntp-e5/E5_train_m-Qwen2.5-7B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-1000"
# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-e5/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-1000"
# MODELS["meta-llama/Meta-Llama-3-8B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Meta-Llama-3-8B-Instruct-mntp-e5/E5_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000"
# MODELS["mistralai/Mistral-7B-Instruct-v0.2"]="/home/siyue/Projects/diffusion_embedder/output/Mistral-7B-Instruct-mntp-e5/E5_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000"

# TASKS=("LEMBNarrativeQARetrieval" "LEMBQMSumRetrieval" "LEMBWikimQARetrieval" "LEMBSummScreenFDRetrieval" "LEMBNeedleRetrieval" "LEMBPasskeyRetrieval")
# TASKS=("LEMBNarrativeQARetrieval")

# for MODEL in "${!MODELS[@]}"; do
#     PEFT="${MODELS[$MODEL]}"
#     MODEL_NAME=$(basename "$MODEL")

#     # Custom output suffix based on model name
#     if [[ "$MODEL_NAME" == "Dream_emb" ]]; then
#         SUFFIX="e5"
#     else
#         SUFFIX="mntp-e5"
#     fi

#     for TASK in "${TASKS[@]}"; do
#         echo "Running $TASK with $MODEL_NAME..."
#         python experiments/mteb_eval_custom.py \
#             --base_model_name_or_path "$MODEL" \
#             --peft_model_name_or_path "$PEFT" \
#             --task_name "$TASK" \
#             --output_dir "results/LEMB/${TASK}/${MODEL_NAME}-${SUFFIX}" \
#             --batch_size 8
#     done
# done

# -----------------------------------------------------------------------------------------#

## train for reasoning-intensive retrieval
# set 4 GPUs

# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Dream_theorem.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Mistral_theorem.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/MetaLlama3_theorem.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Qwen2_theorem.json

## test for reasoning-intensive retrieval
# set 4 GPUs

declare -A MODELS
# MODELS["Qwen/Qwen2.5-7B-Instruct"]=""
# MODELS["siyue/Dream_emb"]=""
MODELS["meta-llama/Meta-Llama-3-8B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Meta-Llama-3-8B-Instruct-mntp-TheoremAug/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-3_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000"
# MODELS["mistralai/Mistral-7B-Instruct-v0.2"]="/home/siyue/Projects/diffusion_embedder/output/Mistral-7B-Instruct-mntp-TheoremAug/E5Mix_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-581"

# TASKS=("BrightTheoremqaTheorems" "BrightTheoremqaQuestions" "BrightAops" "BrightLeetcode")
TASKS=("BrightTheoremqaTheorems")

for MODEL in "${!MODELS[@]}"; do
    PEFT="${MODELS[$MODEL]}"
    MODEL_NAME=$(basename "$MODEL")

    # Custom output suffix based on model name
    if [[ "$MODEL_NAME" == "Dream_emb" ]]; then
        SUFFIX="TheoremAug"
    else
        SUFFIX="mntp-TheoremAug"
    fi

    for TASK in "${TASKS[@]}"; do
        echo "Running $TASK with $MODEL_NAME..."
        python experiments/mteb_eval_custom.py \
            --base_model_name_or_path "$MODEL" \
            --peft_model_name_or_path "$PEFT" \
            --task_name "$TASK" \
            --output_dir "results/BRIGHT/${TASK}/${MODEL_NAME}-${SUFFIX}" \
            --batch_size 16
    done
done





