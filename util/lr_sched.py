# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.WARMUP_EPOCHS:
        lr = cfg.LR * epoch / cfg.WARMUP_EPOCHS
    else:
        lr = cfg.MIN_LR + (cfg.LR - cfg.MIN_LR) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - cfg.WARMUP_EPOCHS) / (cfg.EPOCHS - cfg.WARMUP_EPOCHS)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
