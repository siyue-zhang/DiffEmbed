import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from llm2vec import LLM2Vec
from ReasonAug.util import load_jsonl
import numpy as np
from sklearn.manifold import TSNE

# Load data
data_path = '/home/siyue/Projects/diffusion_embedder/ReasonAug/ReasonAug.jsonl'
data = load_jsonl(data_path)

# Define instances to process
instances = ["Vieta's Formulas", "Pigeonhole Principle", "Euler's Identity", "Central Limit Theorem",
             "Two Pointers", "N-Queens Problem", "Sweep Line Algorithm", "Kahn's Algorithm"]

# Load model
# l2v = LLM2Vec.from_pretrained(
#     "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
#     # peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
#     # peft_model_name_or_path="/home/siyue001/Projects/llm2vec_reason_dream/output/simcse/Meta-Llama-3-8B-Instruct-mntp-simcse-ReasonAug-4omini-all-v0.2/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-32_l-3000_bidirectional-True_e-3_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-800",
#     device_map="cuda" if torch.cuda.is_available() else "cpu",
#     torch_dtype=torch.bfloat16,
# )
l2v = LLM2Vec.from_pretrained(
    "siyue/Dream_emb",
    peft_model_name_or_path="/home/siyue/Projects/diffusion_embedder/output/!Dream-ReasonAug/E5Mix_train_m-Dream_emb_p-mean_b-16_l-4096_bidirectional-True_e-1_s-42_w-100_lr-0.0001_lora_r-16/checkpoint-680",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

# Collect and encode data for each instance
all_embeddings = []
labels = []

for instance in instances:
    group = [d['positive_document'] for d in data if d['task_type'] in ['p2ps','ps2ps'] and d['instance'] == instance]
    if not group:
        print(f"Warning: No data found for instance '{instance}'")
        continue
    reps = l2v.encode(group)
    reps = torch.nn.functional.normalize(reps, p=2, dim=1)
    all_embeddings.append(reps.cpu().numpy())
    labels.extend([instance] * len(reps))

# Combine all embeddings
all_embeddings = np.concatenate(all_embeddings, axis=0)

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
embeddings_2d = tsne.fit_transform(all_embeddings)

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm

# Set global font size
rcParams.update({
    'font.size': 20,           # base font size
    'axes.titlesize': 20,      # title
    'axes.labelsize': 20,      # x and y labels
    'xtick.labelsize': 16,     # x ticks
    'ytick.labelsize': 16,     # y ticks
    'legend.fontsize': 18,     # legend
})

# Plot
plt.figure(figsize=(12, 6))
colors = cm.rainbow(np.linspace(0, 1, len(instances)))

plt.tick_params(
    axis='both',       # Apply to both axes
    which='both',      # Apply to major and minor ticks
    bottom=False,      # Hide bottom ticks
    top=False,         # Hide top ticks (for x)
    left=False,        # Hide left ticks
    right=False,       # Hide right ticks (for y)
    labelbottom=False, # Hide x tick labels
    labelleft=False    # Hide y tick labels
)

for i, instance in enumerate(instances):
    idxs = [j for j, label in enumerate(labels) if label == instance]
    if not idxs:
        continue
    plt.scatter(
        embeddings_2d[idxs, 0],
        embeddings_2d[idxs, 1],
        label=instance,
        alpha=0.9,
        color=colors[i],
        s=100  # Marker size
    )

# plt.title("t-SNE of LLM2Vec Embeddings")
# plt.legend()
# plt.grid(True)
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()
plt.savefig("finetuned_tsne_plot.pdf", dpi=300)
plt.show()







