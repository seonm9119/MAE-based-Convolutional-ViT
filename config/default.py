from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as op
import yaml
from yacs.config import CfgNode as CN

_C = CN()
_C.ENCODER = ''
_C.DATA_PATH = ''
_C.NAME = ''
_C.PIN_MEM = True
_C.NO_PIN_MEM = True
_C.NUM_WORKERS = 10
_C.DEVICE = 'cuda'
_C.SEED = 0
_C.START_EPOCHS = 0
_C.CHECKPOINT = './save/checkpoint.pth'
_C.RESUME = ''
_C.IMG_SIZE = (224, 7)
_C.N_CLASSES = 1000
_C.RATIO = 0.75
_C.MASK = 'rand'

# Distributed training parameters
_C.DDP = CN()
_C.DDP.WORLD_SIZE = 1
_C.DDP.LOCAL_RANK = -1
_C.DDP.DIST_ON_ITP = False
_C.DDP.DIST_URL = 'env://'
_C.DDP.RANK = 1
_C.DDP.GPU = 0
_C.DDP.DISTRIBUTED = False
_C.DDP.DIST_BACKEND = 'nccl'

# Train parameters
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BATCH_PER_GPU = 16
_C.TRAIN.LR = None
_C.TRAIN.BLR = 1e-3
_C.TRAIN.MIN_LR = 0.
_C.TRAIN.WARMUP_EPOCHS = 10
_C.TRAIN.OUTPUT_DIR = './save'
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.RESUME = ''
_C.TRAIN.PRINT_FREQ = 50

# Linear parameters
_C.LINPROBE = CN()
_C.LINPROBE.EPOCHS = 50
_C.LINPROBE.BATCH_PER_GPU = 256 #16
_C.LINPROBE.LR = None
_C.LINPROBE.BLR = 1e-3
_C.LINPROBE.MIN_LR = 0.
_C.LINPROBE.WARMUP_EPOCHS = 10
_C.LINPROBE.OUTPUT_DIR = './lin_output_dir'
_C.LINPROBE.EVAL = False
_C.LINPROBE.RESUME = ''
_C.LINPROBE.PRINT_FREQ = 50

# Finetune parameters
_C.FINETUNE = CN()
_C.FINETUNE.EPOCHS = 100
_C.FINETUNE.BATCH_PER_GPU = 128
_C.FINETUNE.LR = None
_C.FINETUNE.BLR = 1e-3
_C.FINETUNE.MIN_LR = 0.
_C.FINETUNE.WARMUP_EPOCHS = 10

_C.FINETUNE.OUTPUT_DIR = './output_dir'
_C.FINETUNE.EVAL = False
_C.FINETUNE.RESUME = ''
_C.FINETUNE.PRINT_FREQ = 50

_C.FINETUNE.AUG = CN()
_C.FINETUNE.AUG.MIXUP = 0.8
_C.FINETUNE.AUG.MIXCUT = 1.0
_C.FINETUNE.AUG.MIXCUT_MINMAX = None
_C.FINETUNE.AUG.MIXUP_PROB = 1.0
_C.FINETUNE.AUG.MIXUP_SWITCH_PROB = 0.5
_C.FINETUNE.AUG.MIXUP_MODE = 'batch'
_C.FINETUNE.AUG.SMOOTHING = 0.1

_C.MODEL= CN(new_allowed=True)


def update_config(config, cfg_file):

    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            update_config(
                config, op.join(op.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()



if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

