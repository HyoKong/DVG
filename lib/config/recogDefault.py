from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from yacs.config import CfgNode as CN

_C = CN()
_C.GPU = (0,1,2,3)

# network
_C.MODEL = CN()
_C.MODEL.WEIGHT = './output/modelDump/LightCNN_29Layers_V2_checkpoint.pth.tar'
_C.MODEL.NUM_CLASSES = 725

# Dataset
_C.DATASET = CN()
_C.DATASET.ROOT = ''    #
_C.DATASET.LIST_PATH = './lib/utils/generateList1.txt'
_C.DATASET.FAKE_NUM = 100000

# Training hyper-parameters
_C.TRAIN = CN()
_C.TRAIN.LR = 1e-3
_C.TRAIN.ADJUST_LR_STEP = 5
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.NUM_WORKERS = 8

_C.TRAIN.PRE_EPOCH = 6
_C.TRAIN.EPOCH = 15

_C.TRAIN.LAMBDA_MMD = 0.01

_C.TRAIN.PRINT_FREQ = 20
_C.TRAIN.SAVE_EPOCH = 1
_C.TRAIN.RESUME = False

# Misc
_C.MISC = CN()
_C.MISC.OUTPUT_PATH = './output/'


def recogUpdateConfig(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    if args.gpu and '-' in args.gpu:
        ids = args.gpu.split('-')
        ids[0] = 0 if not ids[0].isdigit() else int(ids[0])
        ids[1] = len(mem_info()) if not ids[1].isdigit() else int(ids[1]) + 1
        cfg.GPU = ','.join(map(lambda x: str(x), list(range(*ids))))
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU
        print('>>> Using GPU: {}'.format(cfg.GPU))

    if args.batchSize:
        cfg.TRAIN.BATCH_SIZE = args.batchSize

    if args.cfg:
        (filepath, tempfilename) = os.path.split(args.cfg)
        (filename, extension) = os.path.splitext(tempfilename)
        cfg.CFG_NAME = filename

    print('Hyper Parameters:')
    for k, v in cfg.items():
        print('{}: {}'.format(k,v))

    cfg.freeze()

