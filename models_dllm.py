import os
import sys
import json
import torch

from llama import Tokenizer
from llama.model_train import ModelArgs, Transformer

def LLaMA2_7B_Dynamic(args, **kwargs):
    llama_model_path = args.llama_model_path
    llama_param_path = args.llama_param_path

    checkpoint = torch.load(os.path.join(llama_model_path, "consolidated.00.pth"), map_location="cpu")

    with open(llama_param_path, "r") as f:
        param = json.load(f)
    
    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=32,
        lora_rank=args.lora_rank,
        dynamic_active_target=args.dynamic_active_target,
        dynamic_start_layer=args.dynamic_start_layer,
        dynamic_router_hdim=args.dynamic_router_hdim,
        dynamic_reserve_initials=args.dynamic_reserve_initials,
        **params
    )
    tokenizer = Tokenizer(model_path=args.tokenizer_path)

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_llama_dynamic = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model_llama_dynamic.load_state_dict(checkpoint, strict=False)

    for name, param in model_llama_dynamic.named_parameters():
        if "lora" in name or "router" in name:
            param.requires_grad = True
            param.data = param.data.float()
        else:
            param.requires_grad = False
    
    return model_llama_dynamic