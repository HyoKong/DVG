from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from yacs.config import CfgNode as CN

_C = CN()
_C.GPU = (0,1,2,3)

# Train Generator
_C.G = CN()

# Generator network
_C.G.MODEL = CN()
_C.G.NET_IP = './output/modelDump/LightCNN_29Layers_V2_checkpoint.pth.tar'

# Dataset
_C.G.DATASET = CN()
_C.G.DATASET.ROOT = '/media/HDD8TB/hanyang/face/CASIA_VIS_NIR/'
_C.G.DATASET.PROTOCOLS = './lib/utils/generateList.txt'

# Training hyper-params
_C.G.TRAIN = CN()
_C.G.TRAIN.LR = 1e-3
_C.G.TRAIN.EPOCH = 360
_C.G.TRAIN.MILESTONE = [120, 240]
_C.G.TRAIN.BATCH_SIZE = 256
_C.G.TRAIN.PRINT_FREQ = 20
_C.G.TRAIN.HDIM = 512
_C.G.TRAIN.LAMBDA_MMD = 10.
_C.G.TRAIN.LAMBDA_IP = 50.
_C.G.TRAIN.LAMBDA_PAIR = 50.
_C.G.TRAIN.LAMBDA_REC = 100.
_C.G.TRAIN.LAMBDA_KL = 10.
_C.G.TRAIN.NUM_WORKERS = 8
_C.G.TRAIN.SAVE_EPOCH = 20
_C.G.TRAIN.RESUME = False

_C.G.TRAIN.REC_LOSS = 'l2'  # option: l1, l2, smoothL1
_C.G.TRAIN.IP_LOSS = 'l2'
_C.G.TRAIN.PAIR_LOSS = 'l2'

_C.G.TEST = CN()
_C.G.TEST.FREQ = 10
_C.G.TEST.IMG_DUMP = './output/imgDump/'

# Train Recognition Network
_C.R = CN()

# Misc
_C.MISC = CN()
_C.MISC.OUTPUT_PATH = './output/'


def updateConfig(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    if args.gpu and '-' in args.gpu:
        ids = args.gpu.split('-')
        ids[0] = 0 if not ids[0].isdigit() else int(ids[0])
        ids[1] = len(mem_info()) if not ids[1].isdigit() else int(ids[1]) + 1
        cfg.GPU = ','.join(map(lambda x: str(x), list(range(*ids))))
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU
        print('>>> Using GPU: {}'.format(cfg.GPU))

    if args.hdim:
        cfg.G.HDIM = args.hdim

    if args.outputPath:
        cfg.MISC.OUTPUT_PATH = args.outputPath

    if args.cfg:
        (filepath, tempfilename) = os.path.split(args.cfg)
        (filename, extension) = os.path.splitext(tempfilename)
        cfg.CFG_NAME = filename

    if args.batchSize:
        cfg.G.TRAIN.BATCH_SIZE = args.batchSize

    print('Hyper Parameters:')
    for k, v in cfg.items():
        print('{}: {}'.format(k, v))

    cfg.freeze()



