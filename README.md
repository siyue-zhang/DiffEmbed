# *Diffusion vs. Autoregressive Language Models: A Text Embedding Perspective* 

[**📖 arXiv**](https://arxiv.org/pdf/2505.15045) | [**🤗 Dataset**](https://huggingface.co/datasets/siyue/ReasonAug) 


Diffusion Embedder is a simple recipe to train text encoders directly from diffusion LMs without the need of any transformation step. Diffusion LMs naturally possess bidirectional attention, which is a perfect fit for text embeddings. 

We demonstrate that Diffusion LM embeddings achieve comparable or better performance than Autoregressive LMs across a range of retrieval tasks (state-of-the-art performance), including instruction-following, long-document, and reasoning-intensive retrieval.

<p align="center">
  <img src="https://github.com/siyue-zhang/diffusion_embedder/blob/master/assets/main.png" width="80%" alt="figure1"/>
</p>

This project is implemented based on the code from [LLM2Vec](https://github.com/McGill-NLP/llm2vec/) project and the [Dream](https://github.com/HKUNLP/Dream) model, big thanks to the authors.

## Installation
To use Diffusion Embedder, you can clone this repository and follow these steps:

1. create conda environment and install pip packages:
```bash
conda create -n diffusion_embedder python=3.10.16
conda activate diffusion_embedder
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

> Hints: you need to install ninja first before installing flash-attn; flash-attn only supports Ampere or later Nvidia GPU architectures.

2. pip install local packages, now you are at root directory:
```bash
cd mteb
pip install -e .  # install mteb benchmark
cd ..
pip install -e .  # install llm2vec
```

3. prepare training data for retriever models
   
- MS MARCO with instructions is automatically available from Huggingface [samaya-ai/msmarco-w-instructions](https://huggingface.co/datasets/samaya-ai/msmarco-w-instructions/viewer/default/train?row=1).
- Public E5 is available at Github [echo-embeddings](https://github.com/jakespringer/echo-embeddings). You need to download it in the `cache` directory.
- ReasonAug is in the `ReasonAug` directory or [HuggingFace](https://huggingface.co/datasets/siyue/ReasonAug).

The directory layout should be like follows (custom is optional):
```
cache
|── custom
└── echo-data
    ├── allnli_split1.jsonl
    ├── allnli_split2.jsonl
    ├── allnli.jsonl
    ├── dureader.jsonl
    ...
```

## ReasonAug

This augmentation dataset for the reasoning-intensive retrieval task (e.g., searching math and physics solutions) is constructed using GPT4o-mini. We prompt the LLM to generate question-solution pairs related to a given list of theorems. Theorems include math, physics, finance, and code. The question-solution pairs for the same theorem are positive document for each other. The question-solution pairs for different theorems are negative document for each other.

<p align="center">
  <img src="https://github.com/siyue-zhang/diffusion_embedder/blob/master/assets/gen.png" width="80%" alt="gen"/>
</p>

## Getting Started

LLM2Vec class is a wrapper on top of HuggingFace models to support enabling bidirectionality in decoder-only LLMs, sequence encoding and pooling operations. We have extended LLM2Vec to support Qwen2.5 7B and [Dream 7B (Diffusion reasoning model)](https://github.com/HKUNLP/Dream).

### Train

We conduct supervised contrastive training for Dream for instruction-following retrieval, long-document retrieval, and reasoning-intensive retrieval. For each task, we will finetune the base language models (e.g., Dream) with one training dataset using LoRA.
- long-document retrieval: [Public E5](https://github.com/jakespringer/echo-embeddings)
- reasoning-intensive retrieval: [ReasonAug](https://github.com/siyue-zhang/diffusion_embedder/tree/master/ReasonAug)
- instruction-following retrieval: [msmarco-w-instructions](https://huggingface.co/datasets/samaya-ai/msmarco-w-instructions/viewer/default/train?row=1)

Dataset statistics are shown below.
<p align="center">
  <img src="https://github.com/siyue-zhang/diffusion_embedder/blob/master/assets/statistics.png" width="80%" alt="table"/>
</p>

To train the Meta-Llama-3-8B model with supervised contrastive learning, run the following command:
```
torchrun --nproc_per_node=4 experiments/run_supervised.py train_configs/supervised/MetaLlama3.json
```
Similarly, to train Dream for instruction-following retrieval:
```
torchrun --nproc_per_node=4 experiments/run_supervised.py train_configs/supervised/Dream_if.json
```
LLM2Vec adopts single-negative in-batch negatives as default. We haved added a RepllamaLoss (mimicing [Tevatron](https://github.com/texttron/tevatron/blob/main/src/tevatron/retriever/tevax/loss.py) loss calculation) to support explict multiple negative documents. This is designed to fit the samples of msmarco-w-instructions.

> Hints: remeber to check the GPU ids set in experiments/run_supervised.py. We use 4 A6000 GPUs for experiments.

### Evaluate

We test the retrieval performance on following benchmarks using MTEB implementations:
- instruction-following retrieval: [FollowIR](https://huggingface.co/jhu-clsp)
- long-document retrieval: [LongEmbed](https://huggingface.co/datasets/dwzhu/LongEmbed)
- reasoning-intensive retrieval: [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT)

To evaluate a trained model, you need to run:
```
python experiments/mteb_eval_custom.py \
--base_model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
--peft_model_name_or_path McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp \
--task_name News21InstructionRetrieval \
--output_dir results/followir/News21InstructionRetrieval/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp \
--batch_size 128
```

> Hints: remeber to check the GPU ids set in experiments/mteb_eval_custom.py.


## Citation

If you use the dataset in your work, please kindly cite the paper:
```
@misc{zhang2025diffusionvsautoregressivelanguage,
      title={Diffusion vs. Autoregressive Language Models: A Text Embedding Perspective}, 
      author={Siyue Zhang and Yilun Zhao and Liyuan Geng and Arman Cohan and Anh Tuan Luu and Chen Zhao},
      year={2025},
      eprint={2505.15045},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.15045}, 
}
```


