from __future__ import annotations

from collections import defaultdict

import datasets
import json

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

# import os
# print(os.getcwd())
# from experiments.mteb_eval_custom import DOMAINS_LONG, DOMAINS
DOMAINS_LONG = [
    # "biology",
    # "earth_science",
    # "economics",
    # "psychology",
    # "robotics",
    "stackoverflow",
    # "sustainable_living",
    # "pony",
]

DOMAINS = DOMAINS_LONG + [
    # "leetcode",
    # "aops",
    # "theoremqa_theorems",
    # "theoremqa_questions",
]

DOMAINS_langs = {split: ["eng-Latn"] for split in DOMAINS}


# EVAL_SPLITS = ["standard", "long"]
EVAL_SPLITS = ["standard"]


class BrightStackOverflow(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightStackOverflow",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description=("Bright retrieval dataset."),
        type="Retrieval",
        category="s2p",
        eval_splits=EVAL_SPLITS,
        eval_langs=DOMAINS_langs,
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        form=["written"],
        domains=["Non-fiction"],
        task_subtypes=["Article retrieval"],
        license="CC-BY-4.0",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation="""
            @misc{BRIGHT,
            title={BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
            author={Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and Sun, Ruoxi and Yoon, Jinsung and Arik, Sercan O and Chen, Danqi and Yu, Tao},
            year={2024},
            }
        """,
    )

    def load_bright_data(
        self,
        path: str,
        domains: list,
        eval_splits: list,
        cache_dir: str = None,
        revision: str = None,
        preproc: bool = False,
        fast_bright: str = None,
    ):
        corpus = {domain: {split: None for split in eval_splits} for domain in DOMAINS}
        queries = {domain: {split: None for split in eval_splits} for domain in DOMAINS}
        relevant_docs = {
            domain: {split: None for split in eval_splits} for domain in DOMAINS
        }

        for domain in domains:
            domain_corpus = datasets.load_dataset(
                path, "documents", split=domain, cache_dir=cache_dir, revision=revision
            )

            if fast_bright != None:
                if fast_bright[-1]=='/':
                    fast_bright = fast_bright[:-1]
                rank1_path = f"{fast_bright}/leetcode_bm25_long_False/score.json"
                with open(rank1_path, 'r') as file:
                    rank1_data = json.load(file)
                corpus_ids = []
                for query_id, doc_ids in rank1_data.items():
                    for doc_id in doc_ids:
                        if doc_id not in corpus_ids:
                            corpus_ids.append(doc_id)
                domain_corpus = domain_corpus.filter(lambda x: x['id'] in corpus_ids)
                print('fast eval only ranks\n', domain_corpus)
            
            examples = datasets.load_dataset(
                path, "examples", split=domain, cache_dir=cache_dir, revision=revision
            )
            if domain in DOMAINS_LONG:
                domain_corpus_long = datasets.load_dataset(
                    path,
                    "long_documents",
                    split=domain,
                    cache_dir=cache_dir,
                    revision=revision,
                )

            # TEMP
            # examples = examples.select(range(10))

            # TEMP
            # import json
            # def load_jsonl(filepath):
            #     data = []
            #     with open(filepath, 'r', encoding='utf-8') as file:
            #         for line in file:
            #             data.append(json.loads(line))
            #     return data
            # file = '/home/siyue/Projects/llm2vec_reason/aops_problems_output.jsonl'
            # file = load_jsonl(file)
            # new_query = []
            # for row in file:
            #     q = row['response']['body']['choices'][0]['message']['content']
            #     new_query.append(q)
            # def modify_query(example):
            #     # Suppose new_query is a list or array of new values for the 'query' column
            #     example['query'] = new_query.pop(0)  # Modify the 'query' column
            #     return example
            # examples = examples.map(modify_query)
            #

            corpus[domain]["standard"] = {
                e["id"]: {"text": e["content"]} for e in domain_corpus
            }
            if domain in DOMAINS_LONG:
                corpus[domain]["long"] = {
                    e["id"]: {"text": e["content"]} for e in domain_corpus_long
                }
            queries[domain]["standard"] = queries[domain]["long"] = {
                e["id"]: e["query"] for e in examples
            }
            relevant_docs[domain]["standard"] = {}
            relevant_docs[domain]["long"] = {}

            for e in examples:
                qid = e["id"]
                gold_ids = e["gold_ids"]
                gold_ids_long = e["gold_ids_long"]
                relevant_docs[domain]["standard"][qid] = defaultdict(dict)
                relevant_docs[domain]["long"][qid] = defaultdict(dict)
                for gid in gold_ids:
                    relevant_docs[domain]["standard"][qid].update({gid: 1})
                for gid in gold_ids_long:
                    relevant_docs[domain]["long"][qid].update({gid: 1})

        corpus = datasets.DatasetDict(corpus)
        queries = datasets.DatasetDict(queries)
        relevant_docs = datasets.DatasetDict(relevant_docs)
        return corpus, queries, relevant_docs

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = self.load_bright_data(
            path=self.metadata_dict["dataset"]["path"],
            domains=DOMAINS,
            eval_splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
            preproc=kwargs.get("preproc", False),
            fast_bright=kwargs.get("fast_bright_root", None),
        )
        self.data_loaded = True