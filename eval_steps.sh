#!/bin/bash

declare -A BASE_MODELS
declare -a CHECKPOINTS=(680 300 500)

# Define base paths for each model
BASE_MODELS["Qwen/Qwen2.5-7B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Qwen2.5-7B-TheoremAug-steps/E5Mix_train_m-Qwen2.5-7B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16"
BASE_MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-TheoremAug-steps/E5Mix_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16"

TASKS=("BrightTheoremqaQuestions")

for MODEL in "${!BASE_MODELS[@]}"; do
    BASE_PATH="${BASE_MODELS[$MODEL]}"
    MODEL_NAME=$(basename "$MODEL")

    for CHECKPOINT in "${CHECKPOINTS[@]}"; do
        PEFT="${BASE_PATH}/checkpoint-${CHECKPOINT}"
        
        for TASK in "${TASKS[@]}"; do
            echo "Running $TASK with ${MODEL_NAME} checkpoint-${CHECKPOINT}..."
            python experiments/mteb_eval_custom.py \
                --base_model_name_or_path "$MODEL" \
                --peft_model_name_or_path "$PEFT" \
                --task_name "$TASK" \
                --output_dir "results/TheoQ/${MODEL_NAME}_checkpoint-${CHECKPOINT}" \
                --batch_size 20
        done
    done
done

