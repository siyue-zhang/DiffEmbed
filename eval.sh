#!/bin/bash

# declare -A MODELS
# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/!Dream-e5/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-1000"

# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-e5-64k/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-4000"

# TASKS=("HotpotQA")

# for MODEL in "${!MODELS[@]}"; do
#     PEFT="${MODELS[$MODEL]}"
#     MODEL_NAME=$(basename "$MODEL")

#     for TASK in "${TASKS[@]}"; do
#         echo "Running $TASK with $MODEL_NAME..."
#         python experiments/mteb_eval_custom.py \
#             --base_model_name_or_path "$MODEL" \
#             --peft_model_name_or_path "$PEFT" \
#             --task_name "$TASK" \
#             --output_dir "results/dream_eng_64k/${TASK}/" \
#             --batch_size 128
#     done
# done




# declare -A MODELS
# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/!Dream-e5/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-1000"

# TASKS=("AmazonCounterfactualClassification" "AmazonPolarityClassification" "AmazonReviewsClassification" "ArguAna" "ArxivClusteringP2P" "ArxivClusteringS2S" "AskUbuntuDupQuestions" "BIOSSES" "Banking77Classification" "BiorxivClusteringP2P" "BiorxivClusteringS2S" "CQADupstackTexRetrieval" "EmotionClassification" "FiQA2018" "HotpotQA" "ImdbClassification" "MSMARCO" "MTOPDomainClassification" "MTOPIntentClassification" "MassiveScenarioClassification" "MedrxivClusteringP2P" "MedrxivClusteringS2S" "MindSmallReranking" "NFCorpus" "NQ" "QuoraRetrieval" "RedditClustering" "RedditClusteringP2P" "SCIDOCS" "SICK-R" "STS12" "STS13" "STS14" "STS15" "STS16" "STS17" "STS22" "STSBenchmark" "SciDocsRR" "SciFact" "SprintDuplicateQuestions" "StackExchangeClustering" "StackExchangeClusteringP2P" "StackOverflowDupQuestions" "SummEval" "TRECCOVID" "Touche2020" "ToxicConversationsClassification" "TweetSentimentExtractionClassification" "TwentyNewsgroupsClustering" "TwitterSemEval2015" "TwitterURLCorpus" "FEVER" "DBPedia" "ClimateFEVER" "MassiveIntentClassification")

# for MODEL in "${!MODELS[@]}"; do
#     PEFT="${MODELS[$MODEL]}"
#     MODEL_NAME=$(basename "$MODEL")

#     for TASK in "${TASKS[@]}"; do
#         echo "Running $TASK with $MODEL_NAME..."
#         python experiments/mteb_eval_custom.py \
#             --base_model_name_or_path "$MODEL" \
#             --peft_model_name_or_path "$PEFT" \
#             --task_name "$TASK" \
#             --output_dir "results/dream_eng/${TASK}/" \
#             --batch_size 16
#     done
# done







declare -A MODELS
## Base direct
# MODELS["Qwen/Qwen2.5-7B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Qwen2.5-7B-Instruct-TheoremAug/E5Mix_train_m-Qwen2.5-7B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/!Dream-TheoremAug/E5Mix_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
# MODELS["meta-llama/Meta-Llama-3-8B-Instruct"]="/home/siyue/Projects/diffusion_embedder/output/Meta-Llama-3-8B-Instruct-mntp-TheoremAug/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
MODELS["mistralai/Mistral-7B-Instruct-v0.2"]="/home/siyue/Projects/diffusion_embedder/output/Mistral-7B-Instruct-TheoremAug/E5Mix_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"

## MNTP
# MODELS["McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"]="/home/siyue/Projects/diffusion_embedder/output/!Meta-Llama-3-8B-Instruct-mntp-TheoremAug/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
# MODELS["McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"]="/home/siyue/Projects/diffusion_embedder/output/!Mistral-7B-Instruct-mntp-TheoremAug/E5Mix_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
# MODELS["siyue/LLM2Vec-Qwen2.5-7B-Instruct-mntp"]="/home/siyue/Projects/diffusion_embedder/output/Qwen2.5-7B-Instruct-mntp-TheoremAug/E5Mix_train_m-Qwen2.5-7B-Instruct_p-mean_b-12_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-907"

## SimCSE
# MODELS["McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"]="/home/siyue/Projects/diffusion_embedder/output/Meta-Llama-3-8B-Instruct-mntp-unsup-simcse-TheoremAug/E5Mix_train_m-LLM2Vec-Meta-Llama-3-8B-Instruct-mntp_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"
# MODELS["McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"]="/home/siyue/Projects/diffusion_embedder/output/Mistral-7B-Instruct-mntp-unsup-simcse-TheoremAug/E5Mix_train_m-LLM2Vec-Mistral-7B-Instruct-v2-mntp_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680"

# TASKS=("BrightTheoremqaTheorems")
# TASKS=("BrightTheoremqaTheorems" "BrightTheoremqaQuestions" "BrightAops" "BrightLeetcode")
# TASKS=("BrightTheoremqaTheorems" "BrightTheoremqaQuestions")
TASKS=("BrightAops")
# TASKS=("BrightTheoremqaTheorems" "BrightLeetcode" "BrightTheoremqaQuestions")

for MODEL in "${!MODELS[@]}"; do
    PEFT="${MODELS[$MODEL]}"
    MODEL_NAME=$(basename "$MODEL")

    SUFFIX="TheoremAug"
    for TASK in "${TASKS[@]}"; do
        echo "Running $TASK with $MODEL_NAME..."
        python experiments/mteb_eval_custom.py \
            --base_model_name_or_path "$MODEL" \
            --peft_model_name_or_path "$PEFT" \
            --task_name "$TASK" \
            --output_dir "results/causal_BRIGHT/${TASK}/${MODEL_NAME}-${SUFFIX}" \
            --batch_size 16 \
            --enable_bidirectional False
    done
done