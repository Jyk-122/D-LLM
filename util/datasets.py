import os
import sys
import json
import numpy as np
import copy
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

sys.path.append("../")
from llama import Tokenizer


def load_alpaca(dataset_path, prompt_dict, max_words, tokenizer):
    examples = []
    labels = []
    example_masks = []
    label_masks = []
    prompts = []
    outputs = []

    with open(dataset_path, 'r') as f:
        datas = json.load(f)
    
    for data in tqdm(datas):
        if data["input"] == "":
            prompt = prompt_dict["prompt_wo_input"].format_map(data)
        else:
            prompt = prompt_dict["prompt_with_input"].format_map(data)
        
        example = prompt + data["output"]

        example, label, example_mask, label_mask = tokenize(example, prompt, max_words, tokenizer)

        examples.append(example)
        labels.append(label)
        example_masks.append(example_mask)
        label_masks.append(label_mask)
        prompts.append(prompt)
        outputs.append(data["output"])

    return examples, labels, example_masks, label_masks, prompts, outputs


def tokenize(example, prompt, max_words, tokenizer):
    example = torch.tensor(tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
    prompt = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
    padding = max_words - example.shape[0]
    if padding > 0:
        example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
    label = copy.deepcopy(example)
    label[: len(prompt)] = -1
    example_mask = example.ge(0)
    label_mask = label.ge(0)
    example[~example_mask] = 0
    label[~label_mask] = 0
    example_mask = example_mask.float()
    label_mask = label_mask.float()

    return example, label, example_mask, label_mask


dataset_handlers = {
    "alpaca": load_alpaca,
}


class InstructionDataset(Dataset):
    def __init__(self, dataset_name, dataset_path, partition, tokenizer_path, max_words, truncation_type="random"):

        dataset_name = dataset_name.lower()

        if dataset_name in dataset_handlers:
            handler = dataset_handlers[dataset_name]
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}!")
        
        dataset_path = os.path.join(dataset_path, dataset_name)

        if partition == "train":
            data_path = os.path.join(dataset_path, "train.json")
        elif partition == "test":
            data_path = os.path.join(dataset_path, "test.json")
        else:
            raise ValueError(f"Invalid partiton type: {partition}")
        
        with open(os.path.join(dataset_path, "prompt.json"), "r") as f:
            prompt_dict = json.load(f)

        self.max_words = max_words
        self.truncation_type = truncation_type
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        print(f"Loading dataset >>> {dataset_name} >>> ......")
        datas = handler(data_path, prompt_dict, max_words, self.tokenizer)
        self.examples, self.labels, self.example_masks, self.label_masks, self.prompts, self.outputs = datas

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        label = self.labels[index]
        example_mask = self.example_masks[index]
        label_mask = self.label_masks[index]
        prompt = self.prompts[index]
        output = self.outputs[index]

        max_words = self.max_words
        label_length = torch.sum(label_mask).item()
        index = 0
        if len(example) > max_words:
            if self.truncation_type == "random":
                left = max(0, len(example) - label_length - max_words + 1)
                right = len(example) - max_words + 1
                index = np.random.randint(left, right, (1))[0]
            elif self.truncation_type == "scratch":
                index = 0
            else:
                raise ValueError(f"Invalid truncation type: {self.truncation_type}")
        
        example = example[index : index + max_words]
        label = label[index : index + max_words]
        example_mask = example_mask[index : index + max_words]
        label_mask = label_mask[index : index + max_words]

        return example, label, example_mask, label_mask, prompt, output