import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
from typing import Any
from mteb import mteb
import json
import torch
from datasets import load_dataset

import numpy as np
from mteb.mteb.models.instructions import task_to_instruction
from mteb.mteb.models.text_formatting_utils import corpus_to_texts

from llm2vec import LLM2Vec

from typing import Any, List, Dict, Union

def llm2vec_instruction(instruction):
    if len(instruction) > 0 and instruction[-1] != ":":
        instruction = instruction.strip(".") + ":"
    return instruction


class LLM2VecWrapper:
    def __init__(self, model=None, task_to_instructions=None):

        self.task_to_instructions = task_to_instructions
        self.model = model

    def encode(
        self,
        sentences: List[str],
        *,
        prompt_name: str = None,
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        if prompt_name is not None:
            instruction = (
                self.task_to_instructions[prompt_name]
                if self.task_to_instructions
                and prompt_name in self.task_to_instructions
                else llm2vec_instruction(task_to_instruction(prompt_name))
            )
        else:
            instruction = ""
        sentences = [[instruction, sentence] for sentence in sentences]
        return self.model.encode(sentences, **kwargs)

    def encode_corpus(
        self,
        corpus: Union[List[Dict[str, str]], Dict[str, List[str]], List[str]],
        prompt_name: str = None,
        **kwargs: Any,
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus, sep=" ")
        sentences = [["", sentence] for sentence in sentences]
        if "request_qid" in kwargs:
            kwargs.pop("request_qid")
        return self.model.encode(sentences, **kwargs)

    def encode_queries(self, queries: List[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, **kwargs)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--peft_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument("--task_name", type=str, default="STS16")
    parser.add_argument("--subset_name", type=str, default="leetcode")
    parser.add_argument(
        "--task_to_instructions_fp",
        type=str,
        default="test_configs/mteb/task_to_instructions.json",
    )
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--preproc", action='store_true', help='A boolean flag (True if present, False if absent)')
    parser.add_argument("--fast_bright_root", type=str, default=None)
    parser.add_argument("--enable_bidirectional", type=str2bool, default=True)

    args = parser.parse_args()

    if args.task_name != 'BrightRetrieval':
        mapping = {
            "BrightBiology":"biology",
            "BrightEconomics":"economics",
            "BrightStackOverflow":"stackoverflow",
            "BrightLeetcode":"leetcode",
            "BrightPony":"pony",
            "BrightAops":"aops",
            "BrightTheoremqaTheorems":"theoremqa_theorems",
            "BrightTheoremqaQuestions":"theoremqa_questions",
        }
        if args.task_name in mapping:
            args.subset_name = mapping[args.task_name]

    task_to_instructions = None
    if args.task_to_instructions_fp is not None:
        with open(args.task_to_instructions_fp, "r") as f:
            task_to_instructions = json.load(f)
    
    enable_bidirectional = args.enable_bidirectional
    if args.base_model_name_or_path in [
        "intfloat/e5-mistral-7b-instruct",
    ]:
        enable_bidirectional = False
    print("enable_bidirectional: ", enable_bidirectional)

    l2v_model = LLM2Vec.from_pretrained(
        args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        enable_bidirectional=enable_bidirectional,
        torch_dtype=torch.bfloat16,
        merge_peft=True,
    )

    model = LLM2VecWrapper(model=l2v_model, task_to_instructions=task_to_instructions)
    tasks = mteb.get_tasks(tasks=[args.task_name])
    evaluation = mteb.MTEB(tasks=tasks)

    if 'Bright' in args.task_name:
        data_examples = load_dataset("xlangai/BRIGHT", "examples")[args.subset_name]
        excluded_ids = data_examples["excluded_ids"]
    else:
        excluded_ids = []

    # topk will cut results, set topk>20
    results = evaluation.run(
        model, 
        output_folder=args.output_dir, 
        save_predictions=True, 
        top_k=200, 
        excluded_ids=excluded_ids, 
        batch_size=args.batch_size, 
        preproc=args.preproc,
        fast_bright_root=args.fast_bright_root,
        )


    # excluded_ids