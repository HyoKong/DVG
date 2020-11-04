from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from yacs.config import CfgNode as CN

_C = CN()
_C.GPU_IDS = (0,1,2,3)

# Train Generator
_C.G = CN()

# Generator network
_C.G.MODEL = CN()
_C.G.NET_IP = ''

# Dataset
_C.G.DATASET = CN()
_C.G.DATASET.ROOT = ''

# Training hyper-params
_C.G.TRAIN = CN()
_C.G.TRAIN.LR = 1e-3
_C.G.TRAIN.EPOCH = 500
_C.G.TRAIN.BATCH_SIZE = 32
_C.G.TRAIN.PRINT_FREQ = 20
_C.G.TRAIN.HDIM = 128
_C.G.TRAIN.LAMBDA_MMD = 10.
_C.G.TRAIN.LAMBDA_IP = 10.
_C.G.TRAIN.LAMBDA_PAIR = 10.

# Train Recognition Network
_C.R = CN()

# Misc
_C.MISC = CN()
_C.MISC.OUTPUT_PATH = './output/'


def updateConfig(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    if args.gpuIds and '-' in args.gpuIds:
        ids = args.gpuIds.split('-')
        ids[0] = 0 if not ids[0].isdigit() else int(ids[0])
        ids[1] = len(mem_info()) if not ids[1].isdigit() else int(ids[1]) + 1
        cfg.GPU_IDS = ','.join(map(lambda x: str(x), list(range(*ids))))
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS
        print('>>> Using GPU: {}'.format(cfg.GPU_IDS))

    if args.hdim:
        cfg.G.HDIM = args.hdim

    if args.outputPath:
        cfg.MISC.OUTPUT_PATH = args.outputPath

    print('Hyper Parameters:')
    for k, v in cfg.items():
        print('{}: {}'.format(k, v))

    cfg.freeze()



