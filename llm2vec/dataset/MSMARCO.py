import random
import tiktoken
import numpy as np
from datasets import load_dataset

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

def gpt2_token_count(text):
    # Load the GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    return len(tokens)

class MSMARCO(Dataset):
    def __init__(
        self,
        dataset_name: str = "msmarco-w-instructions",
        split: str = "train",
        file_path: str = None,
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        aug_file_path: str = None,
        domain: str = 'all',
        task: str = 'all',
        add_e5: bool = False
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.train_n_passages = 16
        self.neg_num = self.train_n_passages - 1

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        logger.info(f"Loading data from {file_path}...")
        # # file path is actually a directory

        # total 980k rows
        assert self.split == 'train'
        dataset = load_dataset(file_path)[self.split]
        # we only use first k rows
        dataset = dataset.select(range(16_000))
        # shuffle rows
        # dataset = dataset.shuffle(seed=42) 

        instruction = "Retrieve text based on user query"
        all_samples = []
        id_ = 0
        
        # Process each example in the dataset
        for example in dataset:
            # Format query with instruction
            query = f"{instruction}; {self.separator}{example['query']}"

            # Get the first positive passage
            for pos in example['positive_passages']:
                pos = pos['title'] + ' ' + pos['text'] if 'title' in pos else pos['text']
                pos = self.separator + pos
                break

            new_negative_passages = []
            for neg in example['new_negatives']:
                text = neg['title'] + ' ' + neg['text'] if 'title' in neg else neg['text']
                new_negative_passages.append(self.separator + text)

            negative_passages = []
            for neg in example['negative_passages']:
                text = neg['title'] + ' ' + neg['text'] if 'title' in neg else neg['text']
                negative_passages.append(self.separator + text)
            
            negatives_first_n = min(3, len(new_negative_passages))
            add_neg_num = self.neg_num - negatives_first_n

            # If we don't have enough negatives, randomly sample with replacement
            if len(negative_passages) < add_neg_num:
                negs = random.choices(negative_passages, k=add_neg_num)
            else:
                # Randomly sample without replacement
                negs = random.sample(negative_passages, k=add_neg_num)
            
            negs = new_negative_passages[:negatives_first_n] + negs
            
            all_samples.append(
                DataSample(
                    id_=id_,
                    query=query,
                    positive=pos,
                    task_name="msmarco-w-instruction",
                    batch_negatives=negs,
                )
            )
            id_ += 1
                
        if self.shuffle_individual_datasets:
            random.shuffle(all_samples)
            logger.info(f"Shuffling samples...")
            
        self.data = all_samples
        logger.info(f"Loaded {len(self.data)} samples.")

        # code for statistics
        query_avg_len = []
        doc_avg_len = []
        num_neg = []

        for d in self.data:
            query_avg_len.append(gpt2_token_count(d.query))
            doc_avg_len.append(gpt2_token_count(d.positive))
            num_neg.append(len(d.batch_negatives))
            for x in d.batch_negatives:
                doc_avg_len.append(gpt2_token_count(x))

        print("query len mean: ", np.mean(query_avg_len))
        print("query len std: ", np.std(query_avg_len, ddof=1)) 
        print("doc len mean: ", np.mean(doc_avg_len))
        print("doc len std: ", np.std(doc_avg_len, ddof=1))
        print("avg number of negatives: ", np.mean(num_neg))

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive]+sample.batch_negatives, label=1.0
            )
        elif self.split == "validation":
            assert False, "msmarco-w-instructions does not have a validation split."
