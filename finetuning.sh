#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH=./models
DATASET_PATH=./datasets
OUTPUT_PATH=./output

torchrun --nproc_per_node 8 --master_port 9900 finetuning.py \
    --model_save_name Llama-7B-dynamic \
    --llama_model_path $MODEL_PATH/models \
    --tokenizer_path $MODEL_PATH/../tokenizer.model \
    --dataset_path $DATASET_PATH \
    --dataset_name alpaca \
    --max_seq_len 1024 \
    --lora_rank 8 \
    --dynamic_active_target 0.5 \
    --dynamic_router_hdim 512 \
    --dynamic_start_layer 2 \
    --dynamic_reserve_initials 2 \
    --lambda_active 5.0 \
    --batch_size 1 \
    --epochs 10 \
    --warmup_epochs 2 \
    --save_freq 1 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir $OUTPUT_PATH