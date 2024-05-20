# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

assert timm.__version__ == "0.3.2" # version check
from torchvision import datasets

import util.misc as misc
from util.datasets import linear_transforms
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from config import config, update_config



from engine_finetune import train_one_epoch, evaluate
from models.models_cvt import get_cls_model



def get_args_parser():
    parser = argparse.ArgumentParser('linear probing', add_help=False)
    parser.add_argument('--cfg', default='./config/cvt-13.yaml', type=str,
                        help='experiment configure file name')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--model', default='', type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser = parser.parse_args()
    return parser


def main():

    misc.init_distributed_mode(config.DDP)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    device = torch.device(config.DEVICE)


    seed = config.SEED + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    train_trans = linear_transforms(config.IMG_SIZE[0], is_train=True)
    val_trans = linear_transforms(config.IMG_SIZE[0],is_train=False)

    dataset_train = datasets.ImageFolder(os.path.join(config.DATA_PATH, 'train'), transform=train_trans)
    dataset_val = datasets.ImageFolder(os.path.join(config.DATA_PATH, 'val'), transform=val_trans)


    if config.DDP.DISTRIBUTED:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks,
                                                            rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    if global_rank == 0 and config.LINPROBE.OUTPUT_DIR is not None and not config.LINPROBE.EVAL:
        os.makedirs(config.LINPROBE.OUTPUT_DIR, exist_ok=True)
        log_writer = SummaryWriter(log_dir=config.LINPROBE.OUTPUT_DIR)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train,
                                                    batch_size=config.LINPROBE.BATCH_PER_GPU,
                                                    num_workers=config.NUM_WORKERS,
                                                    pin_memory=config.PIN_MEM,
                                                    drop_last=True)

    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val,
                                                  batch_size=config.LINPROBE.BATCH_PER_GPU,
                                                  num_workers=config.NUM_WORKERS,
                                                  pin_memory=config.PIN_MEM,
                                                  drop_last=False)



    model = get_cls_model(config.MODEL)

    if config.CHECKPOINT and not config.LINPROBE.EVAL:
        checkpoint = torch.load(config.CHECKPOINT, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % config.CHECKPOINT)
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model, strict=False)

    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = config.LINPROBE.BATCH_PER_GPU * misc.get_world_size()
    if config.LINPROBE.LR is None:  # only base_lr is specified
        config.defrost()
        config.LINPROBE.LR = config.LINPROBE.BLR * eff_batch_size / 256
        config.freeze()





    if config.DDP.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.DDP.GPU])
        model_without_ddp = model.module


    optimizer = torch.optim.AdamW(model_without_ddp.head.parameters(), lr=config.LINPROBE.LR)

    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()


    if config.LINPROBE.EVAL:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)
        
    if config.LINPROBE.RESUME:
        misc.load_model(config, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler) 

    print(f"Start training for {config.LINPROBE.EPOCHS} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(config.START_EPOCHS, config.LINPROBE.EPOCHS):
        if config.DDP.DISTRIBUTED:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train,
                                      optimizer, device, epoch, loss_scaler,
                                      None,
                                      log_writer=log_writer, cfg=config.LINPROBE)
        
        if epoch % config.LINPROBE.PRINT_FREQ == 0 or epoch + 1 == config.LINPROBE.EPOCHS:
            misc.save_model(cfg=config.LINPROBE, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'Max accuracy': max_accuracy,
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if config.LINPROBE.OUTPUT_DIR and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(config.LINPROBE.OUTPUT_DIR, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args_parser()
    update_config(config, args.cfg)

    config.defrost()
    config.ENCODER = args.model
    config.LINPROBE.OUTPUT_DIR = os.path.join(config.ENCODER, config.LINPROBE.OUTPUT_DIR)
    config.CHECKPOINT = os.path.join(config.ENCODER, config.CHECKPOINT)
    config.freeze()

    if config.LINPROBE.OUTPUT_DIR:
        Path(config.LINPROBE.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    main()
