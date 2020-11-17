import os
import time
import argparse
import numpy as np
from PIL import Image
import _initPaths

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable

from model import defineG, defineIP
from utils.utils import loadModel

from tqdm import tqdm


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0,1,2,3')
    parser.add_argument('--hdim', default=1024)
    # parser.add_argument('--encVisPath', default='./output/modelDump/recL2.ipL1.pairL1_0749_encVis.pth', help='path of the trained VIS encoder')
    # parser.add_argument('--encNirPath', default='./output/modelDump/recL2.ipL1.pairL1_0749_encNir.pth', help='path of the trained NIR encoder')
    parser.add_argument('--netGPath', default='./output/modelDump/recL2.ipL1.pairL1_0749_netG.pth', help='path of the trained decoder')
    parser.add_argument('--outVisPath', default='./genResults/recL2.ipL1.pairL1.Vis/', help='output path of vis images')
    parser.add_argument('--outNirPath', default='./genResults/recL2.ipL1.pairL1.Nir/', help='output path of nir images')
    return parser.parse_args()


def main():
    args = parseArgs()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not os.path.exists(args.outVisPath):
        os.makedirs(args.outVisPath)
    if not os.path.exists(args.outNirPath):
        os.makedirs(args.outNirPath)

    # Generator
    encVis, encNir, netG = defineG(args.hdim)
    # ckpt = torch.load(args.encVisPath)
    # encVis.load_state_dict(ckpt['model'])
    # ckpt = torch.load(args.encNirPath)
    # encNir.load_state_dict(ckpt['model'])
    ckpt = torch.load(args.netGPath)
    netG.load_state_dict(ckpt['model'])
    netG.eval()

    num = 0

    for n in range(1000):
        noise = torch.zeros(100, args.hdim).normal_(0, 1)
        noise = torch.cat((noise, noise), dim=1)
        noise = Variable(noise).cuda()

        fake = netG(noise)

        nir = fake[:, 0:3, :, :].data.cpu().numpy()
        vis = fake[:, 4:6, :, :].data.cpu().numpy()

        for i in range(nir.shape[0]):
            num += 1
            saveImg = nir[i, :, :, :]
            saveImg = np.transpose((255 * saveImg).astype('uint8'), (1, 2, 0))
            output = Image.fromarray(saveImg)
            saveName = str(num) + '.jpg'
            output.save(os.path.join(args.outNirPath, saveName))

            saveImg = vis[i, :, :, :]
            saveImg = np.transpose((255 * saveImg).astype('uint8'), (1, 2, 0))
            output = Image.fromarray(saveImg)
            saveName = str(num) + '.jpg'
            output.save(os.path.join(args.outVisPath, saveName))

        print(num)


if __name__ == '__main__':
    main()
