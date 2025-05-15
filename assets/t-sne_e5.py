import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from llm2vec import LLM2Vec
from TheoremAug.util import load_jsonl
import numpy as np
from sklearn.manifold import TSNE


import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')
model = model.to(device)

max_length = 4096
# Tokenize the input texts


# Load data
data_path = '/home/siyue/Projects/diffusion_embedder/TheoremAug/TheoremAug.jsonl'
data = load_jsonl(data_path)

# Define instances to process
instances = ["Vieta's Formulas", "Pigeonhole Principle", "Euler's Identity", "Central Limit Theorem",
             "Two Pointers", "N-Queens Problem", "Sweep Line Algorithm", "Kahn's Algorithm"]


all_embeddings = []
labels = []

for instance in instances:
    group = [d['positive_document'] for d in data if d['task_type'] in ['p2ps','ps2ps'] and d['instance'] == instance]
    if not group:
        print(f"Warning: No data found for instance '{instance}'")
        continue

    batch_dict = tokenizer(group, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    # Move batch to GPU
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    with torch.no_grad():  # Add this for inference
        outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    reps = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    all_embeddings.append(reps.cpu().numpy())  # Move back to CPU before converting to numpy
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
    points = embeddings_2d[idxs]
    plt.scatter(
        embeddings_2d[idxs, 0],
        embeddings_2d[idxs, 1],
        label=instance,
        alpha=0.9,
        color=colors[i],
        s=100  # Marker size
    )


    # # Add text labels for each point
    # for point in points:
    #     plt.annotate(
    #         instance,
    #         (point[0], point[1]),
    #         xytext=(5, 5),  # Small offset from the point
    #         textcoords='offset points',
    #         fontsize=12,
    #         color=colors[i],
    #         alpha=0.8
    #     )

# plt.title("t-SNE of LLM2Vec Embeddings")
# plt.legend()
# plt.grid(True)
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()
plt.savefig("e5_tsne_plot.pdf", dpi=300)
plt.show()







