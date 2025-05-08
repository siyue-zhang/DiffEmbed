import torch

from torch import nn
from transformers import PreTrainedModel
from transformers.utils import logging
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from .utils import is_transformers_attn_greater_or_equal_4_43_1
from dream.modeling_dream import (
    DreamModel,
    DreamConfig,
    DreamDecoderLayer,
    DreamAttention,
    DreamSdpaAttention,
    DreamMLP,
    DreamRMSNorm,
    DreamRotaryEmbedding
)

from peft import PeftModel
from typing import Optional, Tuple

logger = logging.get_logger(__name__)

class ModifiedDreamAttention(DreamAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Convert attention mask to same dtype as hidden states
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings, 
        )

class ModifiedDreamSdpaAttention(DreamSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # Convert attention mask to same dtype as hidden states
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            # Ensure attention mask has correct dimensions
            if attention_mask.dim() == 2:
                # Convert [batch_size, seq_length] -> [batch_size, 1, seq_length, seq_length]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.expand(
                    hidden_states.size(0),  # batch size
                    1,                      # num heads
                    hidden_states.size(1),  # seq length
                    hidden_states.size(1)   # seq length
                )
        
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

DREAM_ATTENTION_CLASSES = {
    "eager": ModifiedDreamAttention,
    "sdpa": ModifiedDreamSdpaAttention,
}

class ModifiedDreamDecoderLayer(DreamDecoderLayer):
    def __init__(self, config: DreamConfig, layer_idx: int):
        # Call parent's init instead of nn.Module.__init__
        super().__init__(config, layer_idx)
        
        self.self_attn = DREAM_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

class DreamBiModel(DreamModel):
    _no_split_modules = ["ModifiedDreamDecoderLayer"]

    def __init__(self, config: DreamConfig):
        if not is_transformers_attn_greater_or_equal_4_43_1():
            raise ValueError(
                "The current implementation requires transformers version >= 4.43.1"
            )
        super().__init__(config)
        self.model.layers = nn.ModuleList(
            [
                ModifiedDreamDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    # def _update_causal_mask(
    #     self,
    #     attention_mask,
    #     input_tensor,
    #     cache_position,
    #     past_key_values: Cache,
    #     output_attentions: bool,
    # ):
    #     if attention_mask is not None and 0.0 in attention_mask:
    #         return attention_mask
    #     return None



