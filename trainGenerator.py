import os
import time
import argparse
import _initPaths

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import config
from config import updateConfig

# from utils.utils import createLogger

from model.generator import Encoder, Decoder
from model import defineG, defineIP


def parseArgs():
    parser = argparse.ArgumentParser(description='Train the generator.')
    parser.add_argument('--cfg', help='Experiments configure file name.', required=True)

    parser.add_argument('--gpuIds', type=str, help='gpu ids')
    parser.add_argument('--workers', type=int, help='Number of workers')
    parser.add_argument('--batchSize', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--printFreq', type=int, help='print frequency')
    parser.add_argument('--hdim', type=int, help='dimension of the latent code')

    parser.add_argument('--outputPath', type=str, help='output dir, including logs, model dump and synthetic images etc.')
    args = parser.parse_args()
    updateConfig(config, args)
    return config


def main():
    global args
    cfg = parseArgs()

    if not os.path.exists(cfg.MISC.OUTPUT_PATH):
        os.makedirs(cfg.MISC.OUTPUT_PATH)

    encoderVis, encoderNir, netG = defineG(hdim=cfg.G.TRAIN.HDIM)
    netIP = defineIP(isTrain=False)

    print('==> Loading pre-trained identity preserving model from {}'.format(cfg.G.NET_IP))
    checkpoint = torch.load(cfg.G.NET_IP)
    pretrainedDict = checkpoint['state_dict']
    modelDict = netIP.state_dict()
    pretrainedDict = {k:v for k,v in pretrainedDict.items() if k in modelDict}
    modelDict.update(pretrainedDict)
    netIP.load_state_dict(modelDict)

    for param in netIP.parameters():
        param.requires_grad = False

    # optimizer
    optimizer = torch.optim.Adam(list(netG.parameters()) + list(encoderVis.parameters()) + list(encoderNir.parameters()), lr=cfg.G.TRAIN.LR)

    # criterion
    L2 = torch.nn.MSELoss().cuda()

    # dataloader
    trainLoader = torch.utils.data.DataLoader(

    )

if __name__ == "__main__":
    main()
