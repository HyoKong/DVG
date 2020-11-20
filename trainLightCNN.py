import os
import time
import argparse
import numpy as np
import _initPaths

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.utils as vutils
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from model import LightCNN_29Layers_v2

from config import recogConfig as config
from config import recogUpdateConfig as updateConfig

from data.recogDataset import ImageList, SeparateImageList, SeparateBatchSampler

from utils.utils import AverageMeter, accuracy, saveRecogModel, adjustLearningRate, mmdLoss, saveRecogOptimizer


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='LightCNN')

    parser.add_argument('--gpu', type=str)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--epoch', type=int)

    parser.add_argument('--batchSize', type=int)
    parser.add_argument('--cfg', required=True)
    args = parser.parse_args()
    updateConfig(config, args)
    return config


# pretrain for the last fc2 parameters.
def preTrain(cfg, valLoader, model, criterion, optim, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    for i, data in enumerate(valLoader):
        input = Variable(data['img'].cuda())
        label = Variable(data['label'].cuda())

        output, _ = model(input)

        loss = criterion(output, label)

        # measure accuracy and record loss.
        prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # back propagation.
        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % cfg.TRAIN.PRINT_FREQ == 0:
            info = '===> Epoch [{:0>3d}][{:3d}/{:3d}] | '.format(epoch, i, len(valLoader))
            info += 'Loss: real ce: {:4.3f} ({:4.3f}) | '.format(losses.val, losses.avg)
            info += 'Prec@1: {:4.3f} ({:4.3f}) Prec@5: {:4.3f} ({:4.3f})'.format(top1.val, top1.avg, top5.val, top5.avg)
            print(info)


def validate(valLoader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    for i, data in enumerate(valLoader):
        input = Variable(data['img'].cuda())
        label = Variable(data['label'].cuda())

        output, _ = model(input)
        loss = criterion(output, label)
        prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        print(i)

    print('\nTest set: Avg loss: {}, Top1: ({}), Top5: ({})\n'.format(losses.avg, top1.avg, top5.avg))
    return top1.avg, top5.avg


def train(cfg, trainLoader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    lossesRealCe = AverageMeter()
    lossesRealMmd = AverageMeter()
    lossesFakeMmd = AverageMeter()

    writer = SummaryWriter(logdir=os.path.join(cfg.MISC.OUTPUT_PATH, 'run', '{}'.format(cfg.CFG_NAME)))

    model.train()

    for i, data in enumerate(trainLoader):
        input = Variable(data['img'].cuda())
        label = Variable(data['label'].cuda())
        domain = Variable(data['domain'].cuda())

        # forward
        output, fc = model(input)

        # select nir and vis data
        idxReal = torch.nonzero(label.data != -1)
        idxReal = Variable(idxReal[:, 0])

        outputReal = torch.index_select(output, dim=0, index=idxReal)
        fcReal = torch.index_select(fc, dim=0, index=idxReal)
        labelReal = torch.index_select(label, 0, idxReal)
        domainReal = torch.index_select(domain, 0, idxReal)

        lossRealCe = criterion(outputReal, labelReal) * cfg.TRAIN.LAMBDA_CE

        # select real data
        idxNirReal = torch.nonzero(domainReal.data != 1)
        idxNirReal = Variable(idxNirReal[:, 0])
        fcNirReal = torch.index_select(fcReal, 0, idxNirReal)

        idxVisReal = torch.nonzero(domainReal.data != 0)
        idxVisReal = Variable(idxVisReal[:, 0])
        fcVisReal = torch.index_select(fcReal, 0, idxVisReal)

        lossRealMmd = cfg.TRAIN.LAMBDA_MMD * mmdLoss(fcVisReal, fcNirReal)

        # select fake data
        idxFake = torch.nonzero(label.data == -1)
        idxFake = Variable(idxFake[:, 0])

        fcFake = torch.index_select(fc, 0, idxFake)
        domainFake = torch.index_select(domain, 0, idxFake)

        # select domain of fake data
        idxNirFake = torch.nonzero(domainFake.data != 1)
        idxNirFake = Variable(idxNirFake[:, 0])
        fcNirFake = torch.index_select(fcFake, 0, idxNirFake)

        idxVisFake = torch.nonzero(domainFake.data != 0)
        idxVisFake = Variable(idxVisFake[:, 0])
        fcVisFake = torch.index_select(fcFake, 0, idxVisFake)

        lossFakeMmd = cfg.TRAIN.LAMBDA_MMD * mmdLoss(fcNirFake, fcVisFake)

        lossHFR = lossRealCe + lossRealMmd + lossFakeMmd
        optimizer.zero_grad()
        # TODO(hanyang): need to retain_graph=True??
        lossHFR.backward(retain_graph=True)
        optimizer.step()

        # measure accuracy and record loss
        lossesRealCe.update(lossRealCe.item(), outputReal.size(0))
        lossesRealMmd.update(lossRealMmd.item(), 1)
        lossesFakeMmd.update(lossFakeMmd.item(), 1)

        prec1, prec5 = accuracy(outputReal.data, labelReal.data, topk=(1, 5))
        top1.update(prec1.item(), outputReal.size(0))
        top5.update(prec5.item(), outputReal.size(0))

        # summary writer
        # writer.add_scalar('loss/cross_entropy', lossesRealCe.avg, epoch)
        # writer.add_scalar('loss/real_mmd', lossesRealMmd.avg, epoch)
        # writer.add_scalar('loss/fake_mmd', lossesFakeMmd.avg, epoch)

        if i % cfg.TRAIN.PRINT_FREQ == 0:
        # if True:
            info = '===> Epoch [{:0>3d}][{:3d}/{:3d}] | '.format(epoch, i, len(trainLoader))
            info += 'Loss: real ce: {:4.6f} ({:4.6f}) real mmd: {:4.6f} ({:4.6f}) fake mmd: {:4.6f} ({:4.6f}) | '.format(
                lossesRealCe.val, lossesRealCe.avg, lossesRealMmd.val, lossesRealMmd.avg, lossesFakeMmd.val, lossesFakeMmd.avg
            )
            info += 'Prec@1 : {:4.3f} ({:4.3f}) Prec@5 : {:4.3f} ({:4.3f})'.format(top1.val, top1.avg, top5.val, top5.avg)
            print(info)


def main():
    global args
    cfg = parseArgs()

    # load pre-trained lightcnn model.
    model = LightCNN_29Layers_v2(num_classes=cfg.MODEL.NUM_CLASSES, training=True)
    model = torch.nn.DataParallel(model).cuda()
    print('==> Load pre-trained lightcnn model from {}'.format(cfg.MODEL.WEIGHT))
    ckpt = torch.load(cfg.MODEL.WEIGHT)
    pretrainedDict = ckpt['state_dict']
    modelDict = model.state_dict()
    pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict and 'fc2' not in k}
    modelDict.update(pretrainedDict)
    model.load_state_dict(modelDict)


    # dataset
    imageDataset = SeparateImageList(cfg.DATASET.ROOT, realListPath=cfg.DATASET.LIST_PATH,
                                     fakeVisPath=cfg.DATASET.FAKE_VIS_PATH,
                                     fakeNirPath=cfg.DATASET.FAKE_NIR_PATH,
                                     fakeTotalNum=cfg.DATASET.FAKE_NUM)
    trainRealIdx, trainFakeIdx = imageDataset.getIdx()
    batchSampler = SeparateBatchSampler(realDataIdx=trainRealIdx, fakeDataIdx=trainFakeIdx, batchSize=cfg.TRAIN.BATCH_SIZE * int(len(cfg.GPU)), ratio=0.5)

    # real and fake(generated) training data.
    trainLoader = torch.utils.data.DataLoader(
        imageDataset,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        batch_sampler=batchSampler
    )

    # real training data.
    # ImageList: dataloader of the real dataset.
    valLoader = torch.utils.data.DataLoader(
        ImageList(root=cfg.DATASET.ROOT, fileListPath=cfg.DATASET.VAL_LIST_PATH),
        batch_size=cfg.TRAIN.BATCH_SIZE* int(len(cfg.GPU)), shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True
    )

    # loss
    ceLoss = nn.CrossEntropyLoss().cuda()

    ''' 
    Stage I: model pretrained for last fc2 parameters
    '''
    paramsPretrained = []
    for name, value in model.named_parameters():
        if 'fc2' in name:
            paramsPretrained += [{'params': value, 'lr': 10 * cfg.TRAIN.LR}]

    optimPretrained = torch.optim.SGD(paramsPretrained, cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    for epoch in range(0, cfg.TRAIN.PRE_EPOCH):
        preTrain(cfg, valLoader, model, ceLoss, optimPretrained, epoch)
        saveRecogModel(cfg, model, epoch, prefix='recog_pretrain_model')
        saveRecogOptimizer(cfg, optimPretrained, epoch, prefix='recog_pretrain_optim')

    '''
    Stage II: model fine tune for the overall network.
    '''
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # initial accuracy.
    # prec1, prec5 = validate(valLoader, model, criterion=ceLoss)

    for epoch in range(cfg.TRAIN.PRE_EPOCH, cfg.TRAIN.EPOCH):
        if epoch != 0 and epoch % cfg.TRAIN.ADJUST_LR_STEP == 0:
            adjustLearningRate(cfg.TRAIN.LR, cfg.TRAIN.ADJUST_LR_STEP, optimizer, epoch)
        train(cfg, trainLoader=trainLoader, model=model, criterion=ceLoss, optimizer=optimizer, epoch=epoch)
        prec1, prec5 = validate(valLoader, model, ceLoss)
        saveRecogModel(cfg, model, epoch, prefix='recog_model')
        saveRecogOptimizer(cfg, optimizer, epoch, prefix='recog_optim')

if __name__ == '__main__':
    main()
