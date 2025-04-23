import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,attn_implementation="sdpa")
    model.eval()
    return model, tokenizer

def get_attention_maps(model, tokenizer, text):
    # Modify the input handling to properly create attention mask
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
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

def visualize_attention(attention_maps, tokens, layer=0, head=0, save_dir='attention_plots'):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get attention matrix for specified layer and head
    attention = attention_maps[layer][0][head].numpy()
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        annot=True,
        fmt='.2f'
    )
    plt.title(f'Attention Matrix (Layer {layer}, Head {head})')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    filename = f'attention_layer{layer}_head{head}.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Model path
    model_path = "/home/siyue/Projects/diffusion_embedder/output/Dream-e5/E5_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-5e-05_lora_r-16/checkpoint-1000"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Example text
    text = "This is a sample text to analyze attention patterns."
    
    # Get attention maps and tokens
    attention_maps, tokens = get_attention_maps(model, tokenizer, text)
    
    # Print model information
    print(f"Number of layers: {len(attention_maps)}")
    print(f"Number of attention heads: {attention_maps[0].shape[1]}")
    print(f"Tokens: {tokens}")
    
    # Save directory for plots
    save_dir = 'attention_plots'
    
    # Visualize only the last layer
    last_layer = len(attention_maps) - 1
    num_heads = attention_maps[0].shape[1]
    
    for head in range(num_heads):
        visualize_attention(attention_maps, tokens, layer=last_layer, head=head, save_dir=save_dir)
        print(f"Saved attention plot for last layer ({last_layer}), head {head}")

if __name__ == "__main__":
    main()