#!/bin/bash

declare -A MODELS
# MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/!Dream-e5/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-1000"

MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-e5-64k/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-4000"

TASKS=("AmazonCounterfactualClassification" "AmazonPolarityClassification" "AmazonReviewsClassification" "ArguAna" "ArxivClusteringP2P" "ArxivClusteringS2S" "AskUbuntuDupQuestions" "BIOSSES" "Banking77Classification" "BiorxivClusteringP2P" "BiorxivClusteringS2S" "ClimateFEVER" "CQADupstackTexRetrieval" "EmotionClassification" "FEVER" "FiQA2018" "HotpotQA" "ImdbClassification" "MSMARCO" "MTOPDomainClassification" "MTOPIntentClassification" "MassiveIntentClassification" "MassiveScenarioClassification" "MedrxivClusteringP2P" "MedrxivClusteringS2S" "MindSmallReranking" "NFCorpus" "NQ" "QuoraRetrieval" "RedditClustering" "RedditClusteringP2P" "SCIDOCS" "SICK-R" "STS12" "STS13" "STS14" "STS15" "STS16" "STS17" "STS22" "STSBenchmark" "SciDocsRR" "SciFact" "SprintDuplicateQuestions" "StackExchangeClustering" "StackExchangeClusteringP2P" "StackOverflowDupQuestions" "SummEval" "TRECCOVID" "Touche2020" "ToxicConversationsClassification" "TweetSentimentExtractionClassification" "TwentyNewsgroupsClustering" "TwitterSemEval2015" "TwitterURLCorpus" "DBPedia")

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
            --batch_size 32
    done
done



declare -A MODELS
MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/!Dream-e5/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-1000"

TASKS=("AmazonCounterfactualClassification" "AmazonPolarityClassification" "AmazonReviewsClassification" "ArguAna" "ArxivClusteringP2P" "ArxivClusteringS2S" "AskUbuntuDupQuestions" "BIOSSES" "Banking77Classification" "BiorxivClusteringP2P" "BiorxivClusteringS2S" "ClimateFEVER" "CQADupstackTexRetrieval" "EmotionClassification" "FEVER" "FiQA2018" "HotpotQA" "ImdbClassification" "MSMARCO" "MTOPDomainClassification" "MTOPIntentClassification" "MassiveIntentClassification" "MassiveScenarioClassification" "MedrxivClusteringP2P" "MedrxivClusteringS2S" "MindSmallReranking" "NFCorpus" "NQ" "QuoraRetrieval" "RedditClustering" "RedditClusteringP2P" "SCIDOCS" "SICK-R" "STS12" "STS13" "STS14" "STS15" "STS16" "STS17" "STS22" "STSBenchmark" "SciDocsRR" "SciFact" "SprintDuplicateQuestions" "StackExchangeClustering" "StackExchangeClusteringP2P" "StackOverflowDupQuestions" "SummEval" "TRECCOVID" "Touche2020" "ToxicConversationsClassification" "TweetSentimentExtractionClassification" "TwentyNewsgroupsClustering" "TwitterSemEval2015" "TwitterURLCorpus" "DBPedia")

for MODEL in "${!MODELS[@]}"; do
    PEFT="${MODELS[$MODEL]}"
    MODEL_NAME=$(basename "$MODEL")

    for TASK in "${TASKS[@]}"; do
        echo "Running $TASK with $MODEL_NAME..."
        python experiments/mteb_eval_custom.py \
            --base_model_name_or_path "$MODEL" \
            --peft_model_name_or_path "$PEFT" \
            --task_name "$TASK" \
            --output_dir "results/dream_eng/${TASK}/" \
            --batch_size 32
    done
done



# -----------------------------------------------------------------------------------------#


declare -A MODELS
MODELS["siyue/Dream_emb"]="/home/siyue/Projects/diffusion_embedder/output/Dream-e5-64k/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-2000"

TASKS=("AmazonCounterfactualClassification" "AmazonPolarityClassification" "AmazonReviewsClassification" "ArguAna" "ArxivClusteringP2P" "ArxivClusteringS2S" "AskUbuntuDupQuestions" "BIOSSES" "Banking77Classification" "BiorxivClusteringP2P" "BiorxivClusteringS2S" "ClimateFEVER" "CQADupstackTexRetrieval" "EmotionClassification" "FEVER" "FiQA2018" "HotpotQA" "ImdbClassification" "MSMARCO" "MTOPDomainClassification" "MTOPIntentClassification" "MassiveIntentClassification" "MassiveScenarioClassification" "MedrxivClusteringP2P" "MedrxivClusteringS2S" "MindSmallReranking" "NFCorpus" "NQ" "QuoraRetrieval" "RedditClustering" "RedditClusteringP2P" "SCIDOCS" "SICK-R" "STS12" "STS13" "STS14" "STS15" "STS16" "STS17" "STS22" "STSBenchmark" "SciDocsRR" "SciFact" "SprintDuplicateQuestions" "StackExchangeClustering" "StackExchangeClusteringP2P" "StackOverflowDupQuestions" "SummEval" "TRECCOVID" "Touche2020" "ToxicConversationsClassification" "TweetSentimentExtractionClassification" "TwentyNewsgroupsClustering" "TwitterSemEval2015" "TwitterURLCorpus" "DBPedia")

for MODEL in "${!MODELS[@]}"; do
    PEFT="${MODELS[$MODEL]}"
    MODEL_NAME=$(basename "$MODEL")

    for TASK in "${TASKS[@]}"; do
        echo "Running $TASK with $MODEL_NAME..."
        python experiments/mteb_eval_custom.py \
            --base_model_name_or_path "$MODEL" \
            --peft_model_name_or_path "$PEFT" \
            --task_name "$TASK" \
            --output_dir "results/dream_eng_32k/${TASK}/" \
            --batch_size 64
    done
done
