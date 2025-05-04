#!/bin/bash

declare -A MODELS
# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/!Dream-e5/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-1000"

MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-e5-64k/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-4000"

TASKS=("CQADupstackTexRetrieval")

for MODEL in "${!MODELS[@]}"; do
    PEFT="${MODELS[$MODEL]}"
    MODEL_NAME=$(basename "$MODEL")

    for TASK in "${TASKS[@]}"; do
        echo "Running $TASK with $MODEL_NAME..."
        python experiments/mteb_eval_custom.py \
            --base_model_name_or_path "$MODEL" \
            --peft_model_name_or_path "$PEFT" \
            --task_name "$TASK" \
            --output_dir "results/dream_eng_64k/${TASK}/" \
            --batch_size 64
    done
done

