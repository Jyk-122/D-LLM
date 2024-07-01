import argparse
import copy
import datetime
import json
import os
import time
from pathlib import Path
from typing import Iterable
import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from models_dllm import LLaMA2_7B_Dynamic
from util.datasets import InstructionDataset
import util.lr_sched as lr_sched
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("D-LLM finetuing", add_help=False)
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--accum_iter", default=1, type=int, help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument("--print_freq", default=10, type=int, help="")
    parser.add_argument("--save_freq", default=1, type=int, help="")

    # Model parameters
    parser.add_argument("--llama_model_path", default="./llama", type=str, help="path of llama model")
    parser.add_argument("--tokenizer_path", default="./tokenizer.model", type=str, help="path of tokenizer model")
    parser.add_argument("--model_save_name", type=str, default="Llama_dynamic", help="")
    
    parser.add_argument("--max_seq_len", type=int, default=512, metavar="LENGTH", help="the maximum sequence length")
    parser.add_argument("--lora_rank", type=int, default=8, help="")
    parser.add_argument("--dynamic_active_target", type=float, default=0.5, help="")
    parser.add_argument("--dynamic_router_hdim", type=int, default=512, help="")
    parser.add_argument("--dynamic_start_layer", type=int, default=8, help="")
    parser.add_argument("--dynamic_reserve_initials", type=int, default=2, help="")
    parser.add_argument("--lambda_active", type=float, default=1.0, help="")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR", help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0")
    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--dataset_path", default="./dataset", type=str, help="dataset path")
    parser.add_argument("--dataset_name", default="alpaca", type=str, help="dataset path")
    parser.add_argument("--output_dir", default="./output", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, (example, label, example_mask, label_mask, prompt, output) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            lambda_active = lr_sched.adjust_lambda_active(data_iter_step / len(data_loader) + epoch, args)

        c_loss, a_loss, active_metric = model(example, label, example_mask, label_mask)
        loss = c_loss + lambda_active * a_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        a_loss_value = a_loss.item()
        active_ratio_value = active_metric["mean_scalar_ratio"].item()

        if not math.isfinite(loss_value):
            with open(os.path.join(args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write("Loss is {} when training at example {}-{}.\n".format(loss_value, prompt, output))
            sys.exit(1)

        loss /= accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(closs=c_loss_value)
        metric_logger.update(aloss=a_loss_value)
        metric_logger.update(act=active_ratio_value)
        metric_logger.update(Lact=lambda_active)
        metric_logger.update(lr=lr)

        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        a_loss_value_reduce = misc.all_reduce_mean(a_loss_value)
        active_ratio_value_reduce = misc.all_reduce_mean(active_ratio_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            step = int(len(data_loader) * epoch + data_iter_step) // accum_iter
            log_writer.add_scalar("Train/c_loss", c_loss_value_reduce, step)
            log_writer.add_scalar("Train/a_loss", a_loss_value_reduce, step)
            log_writer.add_scalar("Train/active_ratio", active_ratio_value_reduce, step)
            log_writer.add_scalar("Train/lambda_active", lambda_active, step)
            log_writer.add_scalar("Train/lr", lr, step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter

    c_loss_value_reduce = 0
    a_loss_value_reduce = 0
    active_ratio_value_reduce = 0
    cnt = 0

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, (example, label, example_mask, label_mask, prompt, output) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        cnt += 1

        with torch.no_grad():
            c_loss, a_loss, active_metric = model(example, label, example_mask, label_mask)

        c_loss_value = c_loss.item()
        a_loss_value = a_loss.item()
        active_ratio_value = active_metric["mean_scalar_ratio"].item()

        if not math.isfinite(c_loss_value):
            with open(os.path.join(args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write("Loss is {} when evaluating at example {}-{}.\n".format(c_loss_value, prompt, output))
            sys.exit(1)

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(aloss=a_loss_value)
        metric_logger.update(act=active_ratio_value)

        c_loss_value_reduce += misc.all_reduce_mean(c_loss_value)
        a_loss_value_reduce += misc.all_reduce_mean(a_loss_value)
        active_ratio_value_reduce += misc.all_reduce_mean(active_ratio_value)
    
    c_loss_value_reduce /= cnt
    a_loss_value_reduce /= cnt
    active_ratio_value_reduce /= cnt

    if log_writer is not None:
        log_writer.add_scalar("Train/c_loss", c_loss_value_reduce, epoch)
        log_writer.add_scalar("Train/a_loss", a_loss_value_reduce, epoch)
        log_writer.add_scalar("Train/active_ratio", active_ratio_value_reduce, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):

    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = InstructionDataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        partition="train",
        tokenizer_path=args.tokenizer_path,
        max_words=args.max_seq_len,
    )
    dataset_val = InstructionDataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        partition="test",
        tokenizer_path=args.tokenizer_path,
        max_words=args.max_seq_len,
    )

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    save_folder_name = f"{args.dataset_name}/{args.model_save_name}_{time.strftime('%Y-%m-%d_%H:%M', time.localtime(time.time()))}"

    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, save_folder_name)
        args.log_dir = os.path.join(args.output_dir, "log")
        args.ckpt_dir = os.path.join(args.output_dir, "ckpt")

        if global_rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.log_dir, exist_ok=True)
            os.makedirs(args.ckpt_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
            with open(os.path.join(args.output_dir, "params.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
        else:
            log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = LLaMA2_7B_Dynamic(args)

    if global_rank == 0:
        if args.output_dir:
            with open(os.path.join(args.output_dir, "model_args.json"), "w") as f:
                json.dump(vars(model.params), f, indent=4)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Training on {args.dataset_name} for {args.epochs} epochs, starts at epoch = {args.start_epoch}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        )

        val_stats = val_one_epoch(
            model, data_loader_val, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        )

        if args.ckpt_dir and ((epoch + 1) % (args.save_freq) == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            **{f"val_{k}": v for k, v in val_stats.items()},
        }

        if args.ckpt_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    
    main(args)
