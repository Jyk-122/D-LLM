# D-LLM(WIP)

This repository is the official implementation of [D-LLM: A Token Adaptive Computing Resource Allocation Strategy for Large Language Models](https://arxiv.org/abs/2030.12345). 

The implementation of algorithm is conducted on LLM [Llama-2](https://github.com/Meta-Llama/llama?tab=readme-ov-file) currently.

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets
Datasets used in our work are all public and available on [Huggingface](https://huggingface.co/datasets). We recommend the following form to organize datasets:
```datasets_form
└─datasets
    ├─dataset_name_1
    |   ├─train.json
    |   ├─test.json
    |   └─prompt.json
    └─dataset_name_2
        ├─train.json
        ├─test.json
        └─prompt.json
```
Here we provide two instruction datasets as templates for reference: Alpaca and PIQA. 


## Training

To train the model(s) in the paper, you should appropriate parameters in `finetuning.sh` including:

> - Path to where you place your LLMs' weights - `MODEL_PATH` and params - `MODEL_PARAM_PATH`
> - Path to where you place your datasets - `DATASET_PATH`
> - Main hyperparameters for model and training.

And then, run this command:

```train
bash finetuning.sh
```

## Inference

To inference based on trained D-LLMs, we provide a chat completion program as a demo. Run the following command to ask D-LLM on your own inputs, for example:

```inference
export CUDA_VISIBLE_DEVICE=0

torchrun --nproc_per_node 1 --master_port 9001 ./example.py \
    --llama_ckpt_dir `/path/to/llama_ckpt` \
    --dynamic_ckpt_dir `/path/to/dllm_ckpt` \
    --model_args_path `/path/to/dllm_params` \
    --tokenizer_path `/path/to/llama_tokenizer` \
    --instructs "['Tell me about the music in 1980s.', 'What is new wave?']"
```

You can list your questions as string list in parameter `instructs`. The program outputs answers for default instructions if you don't use `instructs`.

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
