import json
import random
import os
import tiktoken
import numpy as np
from collections import defaultdict
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

def get_doc_and_ids(doc_pairs):
    doc_ids = []
    documents = []
    for dp in doc_pairs:
        doc_ids.append(str(dp['id']))
        documents.append(dp['content'])
    return documents, doc_ids
    
def process_pos_id2doc(entry, id2doc):
    pos_docs = entry["pos"]
    res = []
    for pos in pos_docs:
        instruction, doc_id = pos[0], pos[1]
        doc = id2doc[doc_id]
        res.append([instruction, doc])
    entry["pos"] = res
    return entry


class ReasonIR(Dataset):
    def __init__(
        self,
        dataset_name: str = "ReasonIR",
        split: str = "train",
        file_path: str = None,
        effective_batch_size: int = 16,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        aug_file_path: str = "reasonir/reasonir-data",
        domain: str = 'all',
        task: str = 'all',
        add_e5: bool = False
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.aug_file_path = aug_file_path
        self.add_e5 = add_e5

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):

        data_map = {}
        all_samples = []
        id_ = 0
        if self.add_e5:
            logger.info(f"Loading E5 data from {file_path}...")
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
            keep = 3_000
            for dataset in data_map:
                if len(data_map[dataset]) > keep:
                    data_map[dataset] = random.sample(data_map[dataset], keep)
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


        logger.info(f"Loading ReasonIR data ...")
        # file path is actually a directory

        assert self.split == 'train'  
        hq_dataset = load_dataset(self.aug_file_path, "hq")
        bright_docs = load_dataset("xlangai/BRIGHT", "documents")
        all_docs = []   
        all_ids = []
        for task in bright_docs.keys():
            docs, ids = get_doc_and_ids(bright_docs[task])
            all_docs.extend(docs)
            all_ids.extend(ids)

        id2doc = {}
        for i in range(len(all_docs)):
            id2doc[all_ids[i]] = all_docs[i]

        hq_dataset = hq_dataset.map(lambda x: process_pos_id2doc(x, id2doc))
        vl_dataset = load_dataset(self.aug_file_path, "vl")

        hq_dataset = hq_dataset[self.split].shuffle(seed=42).select(range(20_000))
        vl_dataset = vl_dataset[self.split].shuffle(seed=42).select(range(10_000))

        # hq_dataset = hq_dataset[self.split].shuffle(seed=42)
        # vl_dataset = vl_dataset[self.split].shuffle(seed=42)

        self.all_samples = defaultdict(lambda: [])
        for data in [hq_dataset, vl_dataset]:
            for example in data:
                instruction = example['query'][0]
                instruction = instruction.strip()
                if len(instruction)>0 and instruction[-1]=='.':
                    instruction=instruction[:-1]
                query = f"{instruction}; " + self.separator + example['query'][1]
                pos = example['pos'][0]
                pos = pos[0] + '; ' + self.separator + pos[1]
                neg = example['neg'][0]
                neg = neg[0] + '; ' + self.separator + neg[1]

                self.all_samples[instruction].append(DataSample(
                            id_=id_,
                            query=query,
                            positive=pos,
                            negative=neg,
                            task_name='reasonir',
                        ))
         
        self.data = []
        for dataset in self.all_samples:
            while len(self.all_samples[dataset])>=self.effective_batch_size:
                popped_items = []
                for _ in range(self.effective_batch_size):
                    random_index = random.randint(0, len(self.all_samples[dataset]) - 1)
                    popped_items.append(self.all_samples[dataset].pop(random_index))
                self.data.append(popped_items)

        
        logger.info(f"Loaded {len(self.data)*self.effective_batch_size} augmented samples.")
        
        # if self.add_e5:
        #     tmp = []
        #     for batch in all_batches:
        #         tmp.append([all_samples[idx] for idx in batch])
        #     self.data += tmp

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
            assert False, "ReasonIR does not have a validation split."
