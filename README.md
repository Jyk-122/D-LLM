# D-LLM(WIP)

This repository is the official implementation of [D-LLM: A Token Adaptive Computing Resource Allocation Strategy for Large Language Models](https://arxiv.org/abs/2030.12345) (will be released soon). 

The implementation of algorithm is conducted on [Llama-2](https://github.com/Meta-Llama/llama?tab=readme-ov-file) currently.

![framework](./assets/framework.png)

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
Here we provide two instruction datasets currently as templates for reference: [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned) and [PIQA](https://huggingface.co/datasets/ybisk/piqa). In repository we provide datas as the demonstration with `test.json` for JSON format and `prompt.json` for prompt.


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

To inference based on trained D-LLMs, we provide a chat completion program as a demo. The instruction is organized in the prompt format of Alpaca in `example.py`. Run the following command to ask D-LLM on your own inputs, for example:

```inference
export CUDA_VISIBLE_DEVICE=0

torchrun --nproc_per_node 1 --master_port 9001 ./example.py \
    --llama_ckpt_dir /path/to/llama_ckpt \
    --dynamic_ckpt_dir /path/to/dllm_ckpt \
    --model_args_path /path/to/dllm_params \
    --tokenizer_path /path/to/llama_tokenizer \
    --instructs "['Tell me about the music in 1980s.', 'What is new wave?']"
```

You can list your questions as string list in parameter `instructs`. The program outputs answers for default instructions if you don't use parameter `instructs`.


## Results

We compare D-LLM with other block-wise pruning methods, including MoD, Shortened-Llama, Ada-Infer. We set the target of pruning ratio as 50% for training. D-LLM achieves siginificant improvements on following instruction finetuning benchmarks. The underlined numbers in the following table indicate the best performance. Details are available in our paper.

| Dataset     | MoD   |       | Sh.Lla.PPL |       | Sh.Lla.Tay  |       | Ada-Inf. |       | D-LLM       |       |
| ----------- | ----- | ----- | ---------- | ----- | ----------- | ----- | -------- | ----- | ----------- | ----- |
| Q&A         | PPL   | FLOPs | PPL        | FLOPs | PPL         | FLOPs | PPL      | FLOPs | PPL         | FLOPs |
| Alpaca      | 10.32 | 0.56  | 7.09       | 0.66  | 7.65        | 0.66  | 319      | 0.65  | <u>6.01</u> | 0.59  |
| SAMSum      | 4.47  | 0.56  | 4.39       | 0.66  | 4.66        | 0.66  | 874      | 0.56  | <u>3.18</u> | 0.55  |
| Math        | Acc   | FLOPs | Acc        | FLOPs | Acc         | FLOPs | Acc      | FLOPs | Acc         | FLOPs |
| GSM8K       | 0.08  | 0.56  | 0.1        | 0.66  | 0.18        | 0.66  | 0.00     | 0.83  | <u>0.29</u> | 0.59  |
| MaWPS       | 0.33  | 0.56  | 0.52       | 0.66  | 0.39        | 0.66  | 0.00     | 0.9   | <u>0.74</u> | 0.56  |
| CommonSense | Acc   | FLOPs | Acc        | FLOPs | Acc         | FLOPs | Acc      | FLOPs | Acc         | FLOPs |
| BoolQ       | 0.64  | 0.56  | 0.67       | 0.66  | 0.73        | 0.66  | 0.71     | 0.61  | <u>0.73</u> | 0.52  |
| PIQA        | 0.49  | 0.56  | 0.76       | 0.66  | 0.83        | 0.66  | 0.55     | 0.63  | <u>0.84</u> | 0.52  |
| SIQA        | 0.58  | 0.56  | 0.75       | 0.66  | 0.81        | 0.66  | 0.80     | 0.64  | <u>0.82</u> | 0.54  |
| OBQA        | 0.42  | 0.56  | 0.63       | 0.66  | <u>0.81</u> | 0.66  | 0.78     | 0.76  | 0.80        | 0.53  |
| MMLU        | 0.28  | 0.56  | 0.47       | 0.66  | 0.53        | 0.66  | 0.41     | 0.6   | <u>0.53</u> | 0.55  |



## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
