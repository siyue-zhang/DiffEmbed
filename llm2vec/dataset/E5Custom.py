import json
import random
import os
import tiktoken
import numpy as np
from collections import defaultdict
from pathlib import Path

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

from datasets import load_dataset

logger = get_logger(__name__, log_level="INFO")

E5_EMBEDDING_PROMPTS = {
    "allnli": [
        "Given a premise, retrieve a hypothesis that is entailed by the premise",
        "Retrieve semantically similar text",
    ],
    # "dureader": "Given a Chinese search query, retrieve web passages that answer the question",
    "eli5_question_answer": "Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum",
    "fever": "Given a claim, retrieve documents that support or refute the claim",
    "hotpot_qa": "Given a multi-hop question, retrieve documents that can help answer the question",
    "miracl": "Given a question, retrieve Wikipedia passages that answer the question",
    "mrtydi": "Given a question, retrieve Wikipedia passages that answer the question",
    "msmarco_passage": "Given a web search query, retrieve relevant passages that answer the query",
    "msmarco_document": "Given a web search query, retrieve relevant documents that answer the query",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question",
    "quora_duplicates": [
        "Given a question, retrieve questions that are semantically equivalent to the given question",
        "Find questions that have the same meaning as the input question",
    ],
    "squad": "Retrieve Wikipedia passages that answer the question",
    # "t2ranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "trivia_qa": "Retrieve Wikipedia passages that answer the question",
}

def gpt2_token_count(text):
    # Load the GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    return len(tokens)


def load_custom_data(custom_dataset):
    """Load custom dataset processed from R-retriever data
    
    Args:
        base_dir: Base directory path containing batch_k folders
    """
    print("Loading custom R-retriever dataset...")
    
    res = []
    base_path = Path("/home/siyue/Projects/diffusion_embedder/cache/custom")
       
    # Load passages from any *_7-passage files in the directory
    for json_file in base_path.rglob("*7-passages"):

        if not json_file.is_file():
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                items = json.load(f)
                     
            for item in items:
                # Extract data following preprocess_ourdata.py logic
                key_question = item['enriched_query']['key_question']
                background = item['enriched_query']['background']
                all_pos = item['pos_passage_meta']["positive_passages"]
                all_neg = item['neg_passage_meta']["negative_passages"]
                
                all_pos = [p for p in all_pos if 'passage_text' in p]
                all_neg = [p for p in all_neg if 'passage_text' in p]
                if len(all_pos) * len(all_neg) == 0:
                    continue

                # Format passages
                def extract(passages):    
                    titles = [p['title'] for p in passages]
                    texts = [p['passage_text'] for p in passages]
                    return [f"{t} | {tt}" for t, tt in zip(titles, texts)]
                
                all_pos = extract(all_pos)
                all_neg = extract(all_neg)
                
                pos = random.choice(all_pos)
                neg = random.choice(all_neg)
                query = random.choice([
                    key_question,
                    key_question + ' ' + background,
                    background + ' ' + key_question
                ])

                example = {
                    "query": ["", query],
                    "pos": [["", pos]],
                    "neg": [["", neg]]
                }
                res.append(example)
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    return res


class E5Custom(Dataset):
    def __init__(
        self,
        dataset_name: str = "E5Custom",
        split: str = "train",
        file_path: str = "cache/echo-data",
        aug_file_path: str = None,
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        domain: str = 'all',
        task: str = 'all',
        add_e5: bool = False
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator

        # NEW
        self.aug_file_path = aug_file_path
        self.domain = domain
        self.task = task
        self.add_e5 = add_e5

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        # logger.info(f"Loading E5 data from {file_path}...")
        # # file path is actually a directory

        data_map = {}
        all_samples = []
        id_ = 0
        if self.add_e5:
            for dataset in E5_EMBEDDING_PROMPTS:
                logger.info(f"Loading dataset {dataset}...")
                if dataset not in data_map:
                    data_map[dataset] = []
                with open(os.path.join(file_path, f"{dataset}.jsonl"), "r") as f:
                    dataset_samples = f.readlines()

                dataset_samples = [json.loads(d) for d in dataset_samples]

                for i, sample in enumerate(dataset_samples):
                    instruction = (
                        E5_EMBEDDING_PROMPTS[dataset]
                        if isinstance(E5_EMBEDDING_PROMPTS[dataset], str)
                        else E5_EMBEDDING_PROMPTS[dataset][i % 2]
                    )
                    query = f"{instruction}; " + self.separator + sample["query"]
                    if dataset in [
                        "allnli_split2",
                        "quora_duplicates_split1",
                        "quora_duplicates_split2",
                    ]:
                        pos = (
                            f"{E5_EMBEDDING_PROMPTS[dataset]}; "
                            + self.separator
                            + sample["positive"]
                        )
                        neg = (
                            f"{E5_EMBEDDING_PROMPTS[dataset]}; "
                            + self.separator
                            + sample["negative"]
                        )
                    else:
                        pos = self.separator + sample["positive"]
                        neg = self.separator + sample["negative"]

                    data_map[dataset].append(id_)

                    all_samples.append(
                        DataSample(
                            id_=id_,
                            query=query,
                            positive=pos,
                            negative=neg,
                            task_name=dataset,
                        )
                    )
                    id_ += 1

            # combine split1 and split2
            new_data_map = {}
            for dataset in data_map:
                new_dataset = dataset.replace("_split1", "").replace("_split2", "")
                if new_dataset not in new_data_map:
                    new_data_map[new_dataset] = []
                new_data_map[new_dataset] += data_map[dataset]
            data_map = new_data_map

            # equalize size for each one
            keep = 3000
            for dataset in data_map:
                if len(data_map[dataset])>keep:
                    data_map[dataset] = random.sample(data_map[dataset],keep)
                # print(dataset, len(data_map[dataset]))

            if self.shuffle_individual_datasets:
                for task, samples in data_map.items():
                    random.shuffle(samples)

            datasets = list(data_map.keys())

            logger.info(
                f"Batching Echo data properly for effective batch size of {self.effective_batch_size}..."
            )
            all_batches = []
            for dataset in datasets:
                dataset_samples = data_map[dataset]
                for i in range(0, len(dataset_samples), self.effective_batch_size):
                    batch = dataset_samples[i : i + self.effective_batch_size]
                    if len(batch) == self.effective_batch_size:
                        all_batches.append(batch)
                    else:
                        logger.info(f"Skip 1 batch for dataset {dataset}.")
            random.shuffle(all_batches)

        ## NEW

        # custom_dataset = load_dataset("ya-ir/rtriever-raw_data")
        # custom_dataset = custom_dataset.shuffle(seed=42).select(range(10000))
        custom_dataset = load_custom_data("")
        custom_dataset = random.sample(custom_dataset,30_000)
        self.all_samples = defaultdict(lambda: [])

        for id_, augment_sample in enumerate(custom_dataset):
            instruction, user_query = augment_sample["query"]
            query = f"{instruction}; " + self.separator + user_query
            pos = self.separator.join(augment_sample["pos"][0])
            neg = self.separator.join(augment_sample["neg"][0])
            self.all_samples[instruction].append(DataSample(
                        id_=id_,
                        query=query,
                        positive=pos,
                        negative=neg,
                        task_name='custom',
                    ))
         
        self.data = []
        for dataset in self.all_samples:
            while len(self.all_samples[dataset])>=self.effective_batch_size:
                popped_items = []
                for _ in range(self.effective_batch_size):
                    random_index = random.randint(0, len(self.all_samples[dataset]) - 1)
                    popped_items.append(self.all_samples[dataset].pop(random_index))
                self.data.append(popped_items)

        random.shuffle(self.data)
        
        logger.info(f"Loaded {len(self.data)*self.effective_batch_size} augmented samples.")
        
        if self.add_e5:
            e5 = random.sample(all_batches, int(30000/self.effective_batch_size))
            tmp = []
            for batch in e5:
                tmp.append([all_samples[idx] for idx in batch])
            e5 = tmp
            self.data += e5


        random.shuffle(self.data)
        self.data = [item for sublist in self.data for item in sublist]

        logger.info(f"Loaded {len(self.data)} samples.")

        # code for statistics
        query_avg_len = []
        doc_avg_len = []

        for d in self.data:
            query_avg_len.append(gpt2_token_count(d.query))
            doc_avg_len.append(gpt2_token_count(d.positive))
            doc_avg_len.append(gpt2_token_count(d.negative))

        print("query len mean: ", np.mean(query_avg_len))
        print("query len std: ", np.std(query_avg_len, ddof=1))
        print("doc len mean: ", np.mean(doc_avg_len))
        print("doc len std: ", np.std(doc_avg_len, ddof=1))
        print("total number of samples: ", len(self.data))


    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )
        elif self.split == "validation":
            assert False, "E5Custom does not have a validation split."
