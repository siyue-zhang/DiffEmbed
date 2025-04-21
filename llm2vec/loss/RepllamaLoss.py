import math
import torch
from torch import nn, Tensor
from typing import List


class RepllamaLoss:
    def __init__(
        self,
        scale_by_dim: bool = False,
    ):
        self.scale_by_dim = scale_by_dim
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_negs: List[Tensor],
    ) -> Tensor:

        batch_size = q_reps.size(0) # 8
        num_negatives = len(d_reps_negs) # 16 - 1 = 15
        embedding_dim = q_reps.size(-1)

        # Scale embeddings by dimension if required
        if self.scale_by_dim:
            scale_factor = 1.0 / math.sqrt(embedding_dim)
            q_reps = q_reps * scale_factor
            d_reps_pos = d_reps_pos * scale_factor
            d_reps_negs = [d_neg * scale_factor for d_neg in d_reps_negs]

        # Compute positive scores using dot product
        pos_scores = torch.bmm(q_reps.unsqueeze(1), d_reps_pos.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, 1)

        # Stack negative embeddings
        d_reps_negs_stacked = torch.stack(d_reps_negs, dim=1)  # Shape: (batch_size, num_negatives, embedding_dim)
        
        # Compute negative scores using batch matrix multiplication
        neg_scores = torch.bmm(q_reps.unsqueeze(1), d_reps_negs_stacked.transpose(2, 1)).squeeze(1)  # Shape: (batch_size, num_negatives)

        # Concatenate positive and negative scores
        scores = torch.cat([pos_scores, neg_scores], dim=1)  # Shape: (batch_size, 1 + num_negatives)

        # Labels: The first column corresponds to the positive document
        labels = torch.zeros(batch_size, dtype=torch.long, device=scores.device)

        # Scale the scores and compute the cross-entropy loss
        loss = self.cross_entropy_loss(scores, labels)

        return loss