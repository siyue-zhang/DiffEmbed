import math
import torch
from torch import nn, Tensor
from typing import List
from .loss_utils import cos_sim, mismatched_sizes_all_gather


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

        batch_size = q_reps.size(0)
        embedding_dim = q_reps.size(-1)

        # if torch.distributed.is_initialized():
        #     # Gather embeddings from all GPUs
        #     full_q_reps = mismatched_sizes_all_gather(q_reps)
        #     full_q_reps = torch.cat(full_q_reps)  # (16, 3584)

        #     full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
        #     full_d_reps_pos = torch.cat(full_d_reps_pos)  # (16, 3584)

        #     # Gather negative embeddings from all GPUs
        #     full_d_reps_negs = []
        #     for neg_tensor in d_reps_negs:  # 15 original negative batches
        #         gathered_neg = mismatched_sizes_all_gather(neg_tensor)
        #         gathered_neg = torch.cat(gathered_neg)  # Shape: (16, 3584)
        #         full_d_reps_negs.append(gathered_neg)

        #     rank = torch.distributed.get_rank()
        #     world_size = torch.distributed.get_world_size()
        #     total_batch_size = batch_size * world_size  # 16
            
        #     # Create reorganized negatives - each negative should contain batch_size samples
        #     reorganized_negs = []
            
        #     # 1. Keep original negative batches (15 batches)
        #     current_idx = rank * batch_size
        #     for neg_tensor in full_d_reps_negs:
        #         reorganized_negs.append(neg_tensor[current_idx:current_idx+batch_size])  # (4, 3584)
            
        #     # 2. Add other GPUs' positive docs as negatives (3 GPUs * 4 samples = 12 batches)
        #     for other_rank in range(world_size):
        #         if other_rank != rank:
        #             start_idx = other_rank * batch_size
        #             pos_docs = full_d_reps_pos[start_idx:start_idx+batch_size]  # (4, 3584)
        #             reorganized_negs.append(pos_docs.expand(batch_size, -1))  # (4, 3584)
            
        #     # 3. Add other GPUs' negative docs (3 GPUs * 4 samples * 15 negs = 180 batches)
        #     for other_rank in range(world_size):
        #         if other_rank != rank:
        #             start_idx = other_rank * batch_size
        #             for neg_tensor in full_d_reps_negs:
        #                 neg_docs = neg_tensor[start_idx:start_idx+batch_size]  # (4, 3584)
        #                 reorganized_negs.append(neg_docs.expand(batch_size, -1))  # (4, 3584)
            
        #     # Update working tensors
        #     q_reps = q_reps  # Keep original queries (4, 3584)
        #     d_reps_pos = d_reps_pos  # Keep original positives (4, 3584)
        #     d_reps_negs = reorganized_negs  # List[Tensor(4, 3584)] of length 207
        
        num_negatives = len(d_reps_negs)


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