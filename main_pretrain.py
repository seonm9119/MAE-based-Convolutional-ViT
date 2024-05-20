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
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


from engine_pretrain import train_one_epoch

from util.datasets import pretrain_transforms
from models.model import get_cls_model
from config import config, update_config
from util.loss import InpaintingLoss
from util.loader import MaskDataset

def get_args_parser():
    parser = argparse.ArgumentParser('pre-training', add_help=False)
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

    transform_train = pretrain_transforms(config, 'normal')

    dataset_train = MaskDataset(os.path.join(config.DATA_PATH, 'train'),
                                transform=transform_train, 
                                mask_mode=config.MASK, mask_ratio=config.RATIO,
                                patch_size=config.IMG_SIZE[1], input_size=config.IMG_SIZE[0],
                                )


    if config.DDP.DISTRIBUTED:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks,
                                                            rank=global_rank, shuffle=True)

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)


    if global_rank == 0 and config.TRAIN.OUTPUT_DIR is not None:
        os.makedirs(config.TRAIN.OUTPUT_DIR, exist_ok=True)
        log_writer = SummaryWriter(log_dir=config.TRAIN.OUTPUT_DIR)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train,
                                                    batch_size=config.TRAIN.BATCH_PER_GPU,
                                                    num_workers=config.NUM_WORKERS,
                                                    pin_memory=config.PIN_MEM,
                                                    drop_last=True)

    # define the model
    model = get_cls_model(config.MODEL)


    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    

    eff_batch_size = config.TRAIN.BATCH_PER_GPU * misc.get_world_size()
    if config.TRAIN.LR is None:  # only base_lr is specified
        config.defrost()
        config.TRAIN.LR = config.TRAIN.BLR * eff_batch_size / 256
        config.freeze()


    if config.DDP.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.DDP.GPU],
                                                          #find_unused_parameters=True,
                                                          )
        model_without_ddp = model.module



    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, config.TRAIN.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(param_groups, lr=config.TRAIN.LR, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    criterion = InpaintingLoss()

    if config.RESUME:
        misc.load_model(config, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)  
    
    print(f"Start training for {config.TRAIN.EPOCHS} epochs")
    start_time = time.time()
    for epoch in range(config.START_EPOCHS, config.TRAIN.EPOCHS):
        if config.DDP.DISTRIBUTED:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, data_loader_train,
                                      optimizer, criterion, device, epoch, loss_scaler,
                                      log_writer=log_writer, cfg=config.TRAIN)

        if epoch % config.TRAIN.PRINT_FREQ == 0 or epoch + 1 == config.TRAIN.EPOCHS:
            misc.save_model(cfg=config.TRAIN, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch}

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(config.TRAIN.OUTPUT_DIR, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args_parser()
    update_config(config, args.cfg)

    config.defrost()
    config.ENCODER = args.model
    config.TRAIN.OUTPUT_DIR = os.path.join(config.ENCODER, config.TRAIN.OUTPUT_DIR)
    config.freeze()

    if config.TRAIN.OUTPUT_DIR:
        Path(config.TRAIN.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    main()

