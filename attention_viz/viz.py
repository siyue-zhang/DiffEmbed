import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from datasets import load_dataset
import seaborn as sns
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,attn_implementation="eager")
    model.eval()
    return model, tokenizer

def get_attention_maps(model, tokenizer, text):
    # Modify the input handling to properly create attention mask
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    print(f"Input sequence length: {inputs['input_ids'].shape[1]}")

    # Create proper attention mask
    attention_mask = inputs['attention_mask']
    # Expand attention mask for transformer attention
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=extended_attention_mask,
            output_attentions=True
        )
    
    attention_maps = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    return attention_maps, tokens


def compute_total_attention_metrics(
    attn_matrix: torch.Tensor,
    long_distance: int = 1000,
    short_window: int = 50
):
    """
    Compute total attention metrics for each token:
    - Long-Range Attention Probability (LRAP)
    - Forward Short-Range Attention Probability (forward SRAP)
    - Backward Short-Range Attention Probability (backward SRAP)

    Each metric is the **mean total probability** a token assigns
    to the respective category of tokens (not normalized by count).

    Parameters:
    -----------
    attn_matrix : torch.Tensor
        Attention matrix of shape [seq_len, seq_len]

    Returns:
    --------
    metrics : dict[str, float]
        Dictionary with:
        - 'lrap': mean total attention to distant tokens
        - 'forward_srap': mean total attention to next `short_window` tokens
        - 'backward_srap': mean total attention to previous `short_window` tokens
    """
    seq_len = attn_matrix.shape[0]
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    distance = j - i  # Shape: [seq_len, seq_len]

    # Create masks
    lrap_mask = (torch.abs(distance) > long_distance).float()
    forward_srap_mask = ((distance > 0) & (distance <= short_window)).float()
    backward_srap_mask = ((distance < 0) & (torch.abs(distance) <= short_window)).float()

    # Apply masks to attention matrix
    lrap = (attn_matrix * lrap_mask).sum(dim=1)
    forward_srap = (attn_matrix * forward_srap_mask).sum(dim=1)
    backward_srap = (attn_matrix * backward_srap_mask).sum(dim=1)

    return {
        'lrap': torch.mean(lrap).item(),
        'forward_srap': torch.mean(forward_srap).item(),
        'backward_srap': torch.mean(backward_srap).item()
    }



def load_all_positive_documents():
    # dataset_names = ["narrativeqa", "qmsum", "2wikimqa", "summ_screen_fd", "passkey", "needle"]
    dataset_names = ["2wikimqa"]

    positive_docs = []

    for name in dataset_names:
        # Load corpus and qrels
        corpus = load_dataset("dwzhu/LongEmbed", name=name, split="corpus")
        qrels = load_dataset("dwzhu/LongEmbed", name=name, split="qrels")

        # Create a set of positive document IDs from qrels
        positive_doc_ids = set()
        for entry in qrels:
            positive_doc_ids.add(entry['doc_id'])

        # Filter corpus for positive documents
        for doc in corpus:
            if doc['doc_id'] in positive_doc_ids:
                positive_docs.append(doc['text'])

    return positive_docs


def main():
    # Model path
    model_path = "/home/siyue/Projects/diffusion_embedder/output/!Dream-e5/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-1000"
    # model_path = "/home/siyue/Projects/diffusion_embedder/output/!Mistral-7B-Instruct-mntp-e5/E5_train_m-Mistral-7B-Instruct-v0.2_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-1000"
    print(model_path)
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    positive_docs = load_all_positive_documents()
    # Example text
    text = positive_docs[0]
    
    # Get attention maps and tokens
    attention_maps, tokens = get_attention_maps(model, tokenizer, text)
    
    # Print model information
    print(f"Number of layers: {len(attention_maps)}")
    print(f"Number of attention heads: {attention_maps[0].shape[1]}")
    print(f"Tokens: {len(tokens)}")
    
    # Visualize only the last layer
    last_layer = len(attention_maps) - 1
    num_heads = attention_maps[0].shape[1]
    
    # Initialize dictionaries to store cumulative scores
    # avg_scores = {
    #     'lrap': 0.0,
    #     'forward_srap': 0.0,
    #     'backward_srap': 0.0
    # }

    # print("\nScores for each attention head:")
    # print("Head\tLRAP\tForward SRAP\tBackward SRAP")
    # print("-" * 50)

    for head in range(num_heads):
        attention = attention_maps[last_layer][0][head]

        def compute_entropy(attn_weights):  # shape: (num_heads, seq_len, seq_len)
            # Small epsilon to avoid log(0)
            eps = 1e-8
            attn_weights = attn_weights + eps
            entropy = -torch.sum(attn_weights * torch.log(attn_weights), dim=-1)  # shape: (num_heads, seq_len)
            return entropy  # can average later

        print(compute_entropy(attention))
    
    assert 1==2

    #     scores = compute_total_attention_metrics(attention)
    #     # Print scores for this head
    #     print(f"{head}\t{scores['lrap']:.4f}\t{scores['forward_srap']:.4f}\t{scores['backward_srap']:.4f}")
    #     # Accumulate scores
    #     for metric in avg_scores:
    #         avg_scores[metric] += scores[metric]
    
    # # Calculate averages
    # for metric in avg_scores:
    #     avg_scores[metric] /= num_heads
    
    # print("\nAverage attention metrics across all heads:")
    # for metric, value in avg_scores.items():
    #     print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()