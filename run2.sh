#!/bin/bash

## train for instruction following retrieval
## set 4 GPUs

# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Mistral_if.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Dream_if.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/MetaLlama3_if.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Qwen2_if.json

# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Mistral_if_.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/MetaLlama3_if_.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Qwen2_if_.json

## test for instruction following retrieval
## set 1 GPU

# declare -A MODELS
## Base direct
# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-msmarco/MSMARCO_train_m-Dream_emb_p-mean_b-128_l-304_bidirectional-True_e-1_s-42_w-20_lr-0.0001_lora_r-32/checkpoint-125"
# MODELS["Qwen/Qwen2.5-7B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Qwen2.5-7B-Instruct-msmarco/MSMARCO_train_m-Qwen2.5-7B-Instruct_p-mean_b-128_l-304_bidirectional-True_e-1_s-42_w-20_lr-0.0001_lora_r-32/checkpoint-125"
# MODELS["meta-llama/Meta-Llama-3-8B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Meta-Llama-3-8B-Instruct-msmarco/MSMARCO_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-128_l-304_bidirectional-True_e-1_s-42_w-20_lr-0.0001_lora_r-32/checkpoint-125"
# MODELS["mistralai/Mistral-7B-Instruct-v0.2"]="/home/siyue/Projects/diffusion_embedder/output/Mistral-7B-Instruct-msmarco/MSMARCO_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-128_l-304_bidirectional-True_e-1_s-42_w-20_lr-0.0001_lora_r-32/checkpoint-125"

# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-e5-60k/E5Custom_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-1e-05_lora_r-16/checkpoint-3750"

## MNTP
# MODELS["McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"]="/home/siyue/Projects/diffusion_embedder/output/Meta-Llama-3-8B-Instruct-mntp-msmarco/MSMARCO_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-128_l-304_bidirectional-True_e-1_s-42_w-20_lr-0.0001_lora_r-32/checkpoint-125"
# MODELS["McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"]="/home/siyue/Projects/diffusion_embedder/output/Mistral-7B-Instruct-mntp-msmarco/MSMARCO_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-128_l-304_bidirectional-True_e-1_s-42_w-20_lr-0.0001_lora_r-32/checkpoint-125"
# MODELS["siyue/LLM2Vec-Qwen2.5-7B-Instruct-mntp"]="/home/siyue/Projects/diffusion_embedder/output/Qwen2.5-7B-Instruct-mntp-msmarco/MSMARCO_train_m-Qwen2.5-7B-Instruct_p-mean_b-128_l-304_bidirectional-True_e-1_s-42_w-20_lr-0.0001_lora_r-32/checkpoint-125"

# TASKS=("News21InstructionRetrieval" "Core17InstructionRetrieval" "Robust04InstructionRetrieval" )
# TASKS=("News21InstructionRetrieval")
# TASKS=("Core17InstructionRetrieval")
# TASKS=("Robust04InstructionRetrieval")

# for MODEL in "${!MODELS[@]}"; do
#     PEFT="${MODELS[$MODEL]}"
#     MODEL_NAME=$(basename "$MODEL")

#     # Custom output suffix based on model name
#     if [[ "$MODEL_NAME" == "Dream_emb" ]]; then
#         SUFFIX="Dream_emb-E5-60k"
#     else
#         SUFFIX="msmarco"
#     fi

#     for TASK in "${TASKS[@]}"; do
#         echo "Running $TASK with $MODEL_NAME..."
#         python experiments/mteb_eval_custom.py \
#             --base_model_name_or_path "$MODEL" \
#             --peft_model_name_or_path "$PEFT" \
#             --task_name "$TASK" \
#             --output_dir "results/Dream_e5_custom_FollowIR/${TASK}/${MODEL_NAME}-${SUFFIX}" \
#             --batch_size 64
#     done
# done

declare -A MODELS
MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-e5-custom/E5Custom_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-1e-05_lora_r-16/checkpoint-3750"
TASKS=("News21InstructionRetrieval" "Core17InstructionRetrieval" "Robust04InstructionRetrieval" )
for MODEL in "${!MODELS[@]}"; do
    PEFT="${MODELS[$MODEL]}"
    MODEL_NAME=$(basename "$MODEL")

    # Custom output suffix based on model name
    if [[ "$MODEL_NAME" == "Dream_emb" ]]; then
        SUFFIX="Dream_emb-Custom"
    else
        SUFFIX="msmarco"
    fi

    for TASK in "${TASKS[@]}"; do
        echo "Running $TASK with $MODEL_NAME..."
        python experiments/mteb_eval_custom.py \
            --base_model_name_or_path "$MODEL" \
            --peft_model_name_or_path "$PEFT" \
            --task_name "$TASK" \
            --output_dir "results/Dream_e5_custom_FollowIR/${TASK}/${MODEL_NAME}-${SUFFIX}" \
            --batch_size 32
    done
done

declare -A MODELS
MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-e5-ReasonIR/ReasonIR_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-1e-05_lora_r-16/checkpoint-3750"
TASKS=("News21InstructionRetrieval" "Core17InstructionRetrieval" "Robust04InstructionRetrieval" )
for MODEL in "${!MODELS[@]}"; do
    PEFT="${MODELS[$MODEL]}"
    MODEL_NAME=$(basename "$MODEL")

    # Custom output suffix based on model name
    if [[ "$MODEL_NAME" == "Dream_emb" ]]; then
        SUFFIX="Dream_emb-ReasonIR"
    else
        SUFFIX="msmarco"
    fi

    for TASK in "${TASKS[@]}"; do
        echo "Running $TASK with $MODEL_NAME..."
        python experiments/mteb_eval_custom.py \
            --base_model_name_or_path "$MODEL" \
            --peft_model_name_or_path "$PEFT" \
            --task_name "$TASK" \
            --output_dir "results/Dream_e5_custom_FollowIR/${TASK}/${MODEL_NAME}-${SUFFIX}" \
            --batch_size 32
    done
done












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
## Base direct
# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-e5/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-1000"
# MODELS["Qwen/Qwen2.5-7B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Qwen2.5-7B-Instruct-e5/E5_train_m-Qwen2.5-7B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000"
# MODELS["meta-llama/Meta-Llama-3-8B-Instruct"]=""
# MODELS["mistralai/Mistral-7B-Instruct-v0.2"]="/home/siyue/Projects/diffusion_embedder/output/Mistral-7B-Instruct-e5/E5_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000"

## MNTP
# MODELS["McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"]="/home/siyue/Projects/diffusion_embedder/output/Meta-Llama-3-8B-Instruct-mntp-e5/E5_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000"
# MODELS["McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"]="/home/siyue/Projects/diffusion_embedder/output/!Mistral-7B-Instruct-mntp-e5/E5_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000"
# MODELS["siyue/LLM2Vec-Qwen2.5-7B-Instruct-mntp"]="/home/siyue/Projects/diffusion_embedder/output/!Qwen2.5-7B-Instruct-mntp-e5/E5_train_m-Qwen2.5-7B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000"

# TASKS=("LEMBNarrativeQARetrieval" "LEMBQMSumRetrieval" "LEMBWikimQARetrieval" "LEMBSummScreenFDRetrieval" "LEMBNeedleRetrieval" "LEMBPasskeyRetrieval")
# TASKS=("LEMBNeedleRetrieval" "LEMBPasskeyRetrieval")
# TASKS=("LEMBQMSumRetrieval" "LEMBWikimQARetrieval" "LEMBSummScreenFDRetrieval")

# for MODEL in "${!MODELS[@]}"; do
#     PEFT="${MODELS[$MODEL]}"
#     MODEL_NAME=$(basename "$MODEL")

#     # Custom output suffix based on model name
#     SUFFIX="e5"

#     for TASK in "${TASKS[@]}"; do
#         echo "Running $TASK with $MODEL_NAME..."
#         python experiments/mteb_eval_custom.py \
#             --base_model_name_or_path "$MODEL" \
#             --peft_model_name_or_path "$PEFT" \
#             --task_name "$TASK" \
#             --output_dir "results/LEMB/${TASK}/${MODEL_NAME}-${SUFFIX}" \
#             --batch_size 4
#     done
# done

# -----------------------------------------------------------------------------------------#

## train for reasoning-intensive retrieval
# set 4 GPUs

# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Dream_theorem.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Mistral_theorem.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/MetaLlama3_theorem.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Qwen2_theorem.json

# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Dream_reason.json

# torchrun --nproc_per_node=4 experiments/run_supervised.py /scratch/sz4651/Projects/diffusion_embedder/train_configs/supervised/Dream_reason.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /scratch/sz4651/Projects/diffusion_embedder/train_configs/supervised/MetaLlama3_reason.json
# torchrun --nproc_per_node=4 experiments/run_supervised.py /scratch/sz4651/Projects/diffusion_embedder/train_configs/supervised/Qwen2_reason.json

## test for reasoning-intensive retrieval
# set 4 GPUs

# declare -A MODELS
## Base direct
# MODELS["Qwen/Qwen2.5-7B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Qwen2.5-7B-Instruct-TheoremAug/E5Mix_train_m-Qwen2.5-7B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/!Dream-TheoremAug/E5Mix_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
# MODELS["meta-llama/Meta-Llama-3-8B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Meta-Llama-3-8B-Instruct-mntp-TheoremAug/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
# MODELS["mistralai/Mistral-7B-Instruct-v0.2"]="/home/siyue/Projects/diffusion_embedder/output/Mistral-7B-Instruct-TheoremAug/E5Mix_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"

# MODELS["siyue/Dream_emb"]="/scratch/sz4651/Projects/diffusion_embedder/output/Dream-ResaonIR-mix/checkpoint-2750"
# MODELS["siyue/Dream_emb"]="/scratch/sz4651/Projects/diffusion_embedder/output/Dream-ReasonIR-mix-large/ReasonIR_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-1e-05_lora_r-16/checkpoint-20033"
# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-e5-ReasonIR/ReasonIR_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-1e-05_lora_r-16/checkpoint-3750"
# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-e5-60k/E5Custom_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-1e-05_lora_r-16/checkpoint-3750"

## MNTP
# MODELS["McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"]="/home/siyue/Projects/diffusion_embedder/output/!Meta-Llama-3-8B-Instruct-mntp-TheoremAug/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
# MODELS["McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"]="/home/siyue/Projects/diffusion_embedder/output/!Mistral-7B-Instruct-mntp-TheoremAug/E5Mix_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
# MODELS["siyue/LLM2Vec-Qwen2.5-7B-Instruct-mntp"]="/home/siyue/Projects/diffusion_embedder/output/Qwen2.5-7B-Instruct-mntp-TheoremAug/E5Mix_train_m-Qwen2.5-7B-Instruct_p-mean_b-12_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-907"

# MODELS["siyue/LLM2Vec-Qwen2.5-7B-Instruct-mntp"]="/scratch/sz4651/Projects/diffusion_embedder/output/Qwen2.5-7B-mix-large/ReasonIR_train_m-Qwen2.5-7B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-1e-05_lora_r-16/checkpoint-20033"

# MODELS["McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"]="/scratch/sz4651/Projects/diffusion_embedder/output/Meta-Llama-3-8B-ReasonIR-mix-large/ReasonIR_train_m-LLM2Vec-Meta-Llama-3-8B-Instruct-mntp_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-1e-05_lora_r-16/checkpoint-20033"

## SimCSE
# MODELS["McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"]="/home/siyue/Projects/diffusion_embedder/output/Meta-Llama-3-8B-Instruct-mntp-unsup-simcse-TheoremAug/E5Mix_train_m-LLM2Vec-Meta-Llama-3-8B-Instruct-mntp_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
# MODELS["McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"]="/home/siyue/Projects/diffusion_embedder/output/Mistral-7B-Instruct-mntp-unsup-simcse-TheoremAug/E5Mix_train_m-LLM2Vec-Mistral-7B-Instruct-v2-mntp_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"

# TASKS=("BrightLeetcode")
# TASKS=("BrightTheoremqaQuestions" "BrightAops" "BrightPony" "BrightLeetcode")
# TASKS=("BrightTheoremqaTheorems" "BrightBiology" "BrightEconomics" "BrightStackOverflow" "BrightEarthScience" "BrightPsychology" "BrightRobotics" "BrightSustainableLiving" "BrightTheoremqaQuestions" "BrightAops" "BrightPony" "BrightLeetcode")
# TASKS=("BrightBiology" "BrightEconomics" "BrightStackOverflow" "BrightTheoremqaTheorems")
# TASKS=("BrightEarthScience" "BrightPsychology" "BrightRobotics" "BrightSustainableLiving")

# for MODEL in "${!MODELS[@]}"; do
#     PEFT="${MODELS[$MODEL]}"
#     MODEL_NAME=$(basename "$MODEL")

#     SUFFIX="TheoremAug"
#     for TASK in "${TASKS[@]}"; do
#         echo "Running $TASK with $MODEL_NAME..."
#         python experiments/mteb_eval_custom.py \
#             --base_model_name_or_path "$MODEL" \
#             --peft_model_name_or_path "$PEFT" \
#             --task_name "$TASK" \
#             --output_dir "results/BRIGHT/${TASK}/${MODEL_NAME}-${SUFFIX}" \
#             --batch_size 16
#     done
# done


# for MODEL in "${!MODELS[@]}"; do
#     PEFT="${MODELS[$MODEL]}"
#     MODEL_NAME=$(basename "$MODEL")

#     SUFFIX="E5-60k"
#     for TASK in "${TASKS[@]}"; do
#         echo "Running $TASK with $MODEL_NAME..."
#         python experiments/mteb_eval_custom.py \
#             --base_model_name_or_path "$MODEL" \
#             --peft_model_name_or_path "$PEFT" \
#             --task_name "$TASK" \
#             --output_dir "results/Dream_e5_custom_BRIGHT/${TASK}/${MODEL_NAME}-${SUFFIX}" \
#             --batch_size 16
#     done
# done


# torchrun --nproc_per_node=4 experiments/run_supervised.py /home/siyue/Projects/diffusion_embedder/train_configs/supervised/Dream_custom.json


# TASKS=("BrightTheoremqaTheorems" "BrightLeetcode" "BrightTheoremqaQuestions")
# # TASKS=("BrightTheoremqaTheorems")

# for TASK in "${TASKS[@]}"; do
#     python experiments/mteb_eval_custom.py \
#         --base_model_name_or_path "intfloat/e5-mistral-7b-instruct" \
#         --task_name "$TASK" \
#         --output_dir "results/BRIGHT/${TASK}/e5-mistral-7b-instruct" \
#         --enable_bidirectional False \
#         --batch_size 24 
# done

