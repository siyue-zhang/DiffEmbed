from llm2vec import LLM2Vec

import json
import logging
import os
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from peft import PeftModel
from torch import Tensor, device, nn

from transformers import (
    AutoModel,
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,
)


# class
    

#     def forward(self, sentence_feature: Dict[str, Tensor]):
#         embed_mask = None
#         if "embed_mask" in sentence_feature:
#             embed_mask = sentence_feature.pop("embed_mask")
#         reps = self.model(**sentence_feature)
#         sentence_feature["embed_mask"] = embed_mask

#         return self.get_pooling(sentence_feature, reps.last_hidden_state)



class FusionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(FusionLayer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, query_embedding, document_embedding):
        # Cross-attention between query and document embeddings
        fused_embedding, _ = self.cross_attention(
            query_embedding.unsqueeze(0),  # Add sequence dimension
            document_embedding.unsqueeze(0),
            document_embedding.unsqueeze(0),
        )
        fused_embedding = fused_embedding.squeeze(0)  # Remove sequence dimension

        # Concatenate and pass through feedforward layer
        fused_embedding = torch.cat([query_embedding, document_embedding], dim=-1)
        fused_embedding = self.feedforward(fused_embedding)
        return fused_embedding
    

class LLM2VecWithDecoder(nn.Module):
    def __init__(
        self,
        encoder: LLM2Vec,
        decoder: AutoModel,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        hidden_size = self.decoder.config["hidden_size"]
        self.fusion_layer = FusionLayer(hidden_size)
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"


    @classmethod
    def from_pretrained_decoder(
        cls,
        encoder,
        base_model_name_or_path,
        peft_model_name_or_path=None,
        merge_peft=False,
        enable_bidirectional=False,
        **kwargs,
    ):
        # pop out decoder args
        keys = ["max_length", ]
        decoder_args = {
            key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None
        }

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(base_model_name_or_path)
        config_class_name = config.__class__.__name__

        model_class = cls._get_model_class(
            config_class_name, enable_bidirectional=enable_bidirectional
        )
        model = model_class.from_pretrained(base_model_name_or_path, **kwargs)

        if os.path.isdir(base_model_name_or_path) and os.path.exists(
            f"{base_model_name_or_path}/config.json"
        ):
            with open(f"{base_model_name_or_path}/config.json", "r") as fIn:
                config_dict = json.load(fIn)
            config = PretrainedConfig.from_dict(config_dict)
            model.config._name_or_path = config._name_or_path

        # For special case where config.json and adapter weights are in the same directory
        if hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(
                model,
                base_model_name_or_path,
            )
            model = model.merge_and_unload()

        if peft_model_name_or_path is not None:
            model = PeftModel.from_pretrained(
                model,
                peft_model_name_or_path,
            )
            if merge_peft:
                model = model.merge_and_unload()

        config = {}
        config_addr = (
            peft_model_name_or_path
            if peft_model_name_or_path is not None
            else base_model_name_or_path
        )
        if os.path.exists(f"{config_addr}/llm2vec_config.json"):
            with open(f"{config_addr}/llm2vec_config.json", "r") as fIn:
                llm2vec_config = json.load(fIn)
            config.update(llm2vec_config)

        for key, value in decoder_args.items():
            config[key] = value

        return cls(encoder=encoder, decoder=model, tokenizer=tokenizer, **config)


    def forward(
        self,
        input_texts: Union[str, List[str]],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        max_length: int = 512,
        num_beams: int = 5,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        """
        Forward pass through the encoder-decoder pipeline.
        Args:
            input_texts: Input text(s) to encode and decode. Can be a single string or a list of strings.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to show a progress bar during encoding.
            max_length: Maximum length of the generated output.
            num_beams: Number of beams for beam search.
            temperature: Sampling temperature.
            top_k: Top-k sampling.
            top_p: Top-p (nucleus) sampling.
        Returns:
            Generated text outputs.
        """
        # Ensure input_texts is a list
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        # Step 1: Encode the input texts into embeddings using LLM2Vec
        embeddings = self.encoder.encode(
            sentences=input_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
            device=self.device,
        )
        features, labels = inputs
        q_reps = self.model(features[0])
        d_reps = self.model(features[1])

        # Step 2: Fuse query and document embeddings
        query_embeddings = embeddings[0]
        document_embeddings = embeddings[1]
        fused_embeddings = self.fusion_layer(query_embeddings, document_embeddings)

        # Step 3: Generate text using the decoder model
        outputs = self.decoder.generate(
            inputs_embeds=fused_embeddings,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )

        # Step 3: Decode the generated token IDs into text
        # generated_texts = self.encoder.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return embeddings, outputs