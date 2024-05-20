# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from config import config

def pretrain_transforms(cfg, mode='normal'):

    if mode == 'cvt':
        timm_cfg = cfg.AUG.TIMM_AUG
        transforms = create_transform(
            input_size=224,
            is_training=True,
            use_prefetcher=False,
            no_aug=False,
            re_prob=timm_cfg.RE_PROB,
            re_mode=timm_cfg.RE_MODE,
            re_count=timm_cfg.RE_COUNT,
            scale=cfg.AUG.SCALE,
            ratio=cfg.AUG.RATIO,
            hflip=timm_cfg.HFLIP,
            vflip=timm_cfg.VFLIP,
            color_jitter=timm_cfg.COLOR_JITTER,
            auto_augment=timm_cfg.AUTO_AUGMENT,
            interpolation=timm_cfg.INTERPOLATION,
            mean=IMAGENET_DEFAULT_MEAN,
            std=[0.229, 0.224, 0.225],
        )

        return transforms

    elif mode == 'normal':
        transforms = T.Compose([T.RandomResizedCrop(cfg.IMG_SIZE[0], scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

        return transforms



def linear_transforms(size, is_train):
    # train transform
    if is_train:
        transform = T.Compose([T.RandomResizedCrop(size, interpolation=InterpolationMode.BICUBIC),
                               T.RandomHorizontalFlip(),
                               T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        return transform

    transform = T.Compose([
        T.Resize(int(1.145*size), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transform


def prediction_transforms(is_train):
    # train transform
    if is_train:
        transform = T.Compose([T.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                               T.RandomHorizontalFlip(),
                               T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        return transform

    transform = T.Compose([
        T.Resize(256, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transform



