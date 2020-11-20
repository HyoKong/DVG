from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import glob

import torch
import torch.optim as optim
import torch.nn.functional as F

class AverageMeter(object):
    '''
    Compute and store the average and current value.
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rgb2gray(img):
    r, g, b = torch.split(img, split_size_or_sections=1, dim=1)
    return torch.mul(r, 0.299) + torch.mul(g, 0.587) + torch.mul(b, 0.114)


def reparameterize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    return eps.mul(std).add_(mu)


def saveModel(cfg, encVis, encNir, netG, epoch):
    prefix = 'encVis'
    modelPath = os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump', '{}_{}_{}.pth'.format(cfg.CFG_NAME, str(epoch).zfill(4), prefix))
    stateDict = {'epoch': epoch, 'model': encVis}
    if not os.path.exists(cfg.MISC.OUTPUT_PATH):
        os.makedirs(cfg.MISC.OUTPUT_PATH)
    if not os.path.exists(os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump')):
        os.makedirs(os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump'))
    torch.save(stateDict, modelPath)

    prefix = 'encNir'
    modelPath = os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump', '{}_{}_{}.pth'.format(cfg.CFG_NAME, str(epoch).zfill(4), prefix))
    stateDict = {'epoch': epoch, 'model': encNir}
    torch.save(stateDict, modelPath)

    prefix = 'netG'
    modelPath = os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump', '{}_{}_{}.pth'.format(cfg.CFG_NAME, str(epoch).zfill(4), prefix))
    stateDict = {'epoch': epoch, 'model': netG}
    torch.save(stateDict, modelPath)

    print('Save checkpoints to {}'.format(os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump')))


def saveOptimizer(cfg, optimizer, epoch):
    checkpointPath = os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump', '{}_{}_optim.pth'.format(cfg.CFG_NAME, str(epoch).zfill(4)))
    stateDict = {'optim': optimizer}
    if not os.path.exists(cfg.MISC.OUTPUT_PATH):
        os.makedirs(cfg.MISC.OUTPUT_PATH)
    if not os.path.exists(os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump')):
        os.makedirs(os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump'))
    torch.save(stateDict, checkpointPath)
    print('Save optimizer to {}.'.format(checkpointPath))


def loadOptimizer(cfg, optimizer):
    optimizerList = glob.glob(os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump', '{}_*.pth'.format(cfg.CFG_NAME)))
    curEpoch = max([int(str(fileName[fileName.find('{}_'.format(cfg.CFG_NAME)) + len('{}_'.format(cfg.CFG_NAME))
                                     :fileName.find('_optim.pth')])) for fileName in optimizerList])
    ckpt = torch.load(os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump', '{}_{}_optim.pth'.format(cfg.CFG_NAME, str(curEpoch).zfill(4))))
    optimizer.load_state_dict(ckpt['optim'])
    print('Load optimizer from {}.'.format(os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump', '{}_{}_optim.pth'.format(cfg.CFG_NAME, str(curEpoch).zfill(4)))))
    return optimizer


def loadModel(cfg, encNir, encVis, netG):
    modelList = glob.glob(os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump', '{}_*.pth'.format(cfg.CFG_NAME)))
    curEpoch = max([int(str(fileName[len('{}_'.format(cfg.CFG_NAME)):len('{}_'.format(cfg.CFG_NAME)) + 4])) for fileName in modelList])

    prefix = 'encVis'
    ckpt = torch.load(os.path.join(cfg.CFG_NAME, 'modelDump', '{}_{}_{}.pth'.format(cfg.CFG_NAME, str(curEpoch).zfill(4), prefix)))
    encVis.load_state_dict(ckpt['model'])
    startEpoch = ckpt['epoch'] + 1

    prefix = 'encNir'
    ckpt = torch.load(os.path.join(cfg.CFG_NAME, 'modelDump', '{}_{}_{}.pth'.format(cfg.CFG_NAME, str(curEpoch).zfill(4), prefix)))
    encNir.load_state_dict(ckpt['model'])

    prefix = 'netG'
    ckpt = torch.load(os.path.join(cfg.CFG_NAME, 'modelDump', '{}_{}_{}.pth'.format(cfg.CFG_NAME, str(curEpoch).zfill(4), prefix)))
    netG.load_state_dict(ckpt['model'])

    return encVis, encNir, netG, startEpoch


def accuracy(output, target, topk=(1,)):
    '''
    Compute the precision@k for the specified values of k.
    :param output: output
    :param target: ground truth
    :param topk:
    :return:
    '''
    maxk = max(topk)
    batchSize = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # transpose
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correctK = correct[:k].view(-1).float().sum(0)
        res.append(correctK.mul_(100. / batchSize))
    return res


def saveRecogModel(cfg, model, epoch, prefix):
    state = {'epoch': epoch, 'state_dict': model.state_dict()}
    modelPath = os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump', '{}_{}_epoch_{:0>4d}.pth'.format(prefix, cfg.CFG_NAME, epoch))
    if not os.path.exists(cfg.MISC.OUTPUT_PATH):
        os.makedirs(cfg.MISC.OUTPUT_PATH)
    if not os.path.exists(os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump')):
        os.makedirs(os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump'))

    torch.save(state, modelPath)
    print('Recognition model saved to {}'.format(modelPath))


def saveRecogOptimizer(cfg, optimizer, epoch, prefix):
    '''
    save recognition optimizer.
    format: recog_{cfg.CFG_NAME}_epoch_{epoch}.pth
    :param cfg:
    :param optimizer:
    :param epoch:
    :param prefix:
    :return:
    '''
    ckptPath = os.path.join(cfg.MISC.OUTPUT_PATH, 'modelDump', '{}_{}_epoch_{:0>4d}.pth'.format(prefix, cfg.CFG_NAME, epoch))
    state = {'optim': optimizer.state_dict()}
    torch.save(state, ckptPath)
    print('Save recognition optimizer to {}'.format(ckptPath))


def adjustLearningRate(lr, step, optimizer, epoch):
    scale = 0.45705051927326
    lr = lr * (scale ** (epoch // step))
    print('Updated LR: {:.6f}'.format(lr))

    for paramGroup in optimizer.param_groups:
        paramGroup['lr'] = paramGroup['lr'] * scale

def mmdLoss(fcNir, fcVis):
    meanFcNir = torch.mean(fcNir, 0)
    meanFcVis = torch.mean(fcVis, 0)
    lossMmd = F.mse_loss(meanFcNir, meanFcVis)
    return lossMmd

def revise_name(img_name):
    '''
    's2\\NIR\\10117\\016.bmp ==> s2\\NIR_128x128\\10117\\016.bmp'
    :param img_name:
    :return:
    '''
    suffix = img_name.split('.')
    if suffix[-1] != 'jpg':
        suffix[-1] = 'jpg'

    img_name = '.'.join(suffix)
    revise_name = img_name.split('\\')  # img_name: s2\NIR\10117\016.bmp
    # revise_name[1] += '_128x128'
    temp = ''
    for i in range(len(revise_name)):
        temp = temp + revise_name[i]
        if i != len(revise_name) - 1:
            temp += '_'
    return temp


