import os
import time
import argparse
import _initPaths

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from datetime import datetime

from config import config
from config import updateConfig

from utils.utils import AverageMeter, reparameterize, rgb2gray, saveModel, saveOptimizer, loadOptimizer, loadModel

from model.generator import Encoder, Decoder
from model import defineG, defineIP

from data import GenDataset

from loss import reconLoss, klLoss


def parseArgs():
    parser = argparse.ArgumentParser(description='Train the generator.')
    parser.add_argument('--cfg', help='Experiments configure file name.', required=True)

    parser.add_argument('--gpu', type=str, help='gpu ids')
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
    netIP = defineIP(isTrain=False, )

    print('==> Loading pre-trained identity preserving model from {}'.format(cfg.G.NET_IP))
    checkpoint = torch.load(cfg.G.NET_IP)
    pretrainedDict = checkpoint['state_dict']
    modelDict = netIP.state_dict()
    pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
    modelDict.update(pretrainedDict)
    netIP.load_state_dict(modelDict)

    for param in netIP.parameters():
        param.requires_grad = False

    # optimizer
    optimizer = torch.optim.Adam(list(netG.parameters()) + list(encoderVis.parameters()) + list(encoderNir.parameters()), lr=cfg.G.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.G.TRAIN.MILESTONE, gamma=0.1, last_epoch=-1)

    # resume
    if cfg.G.TRAIN.RESUME:
        encoderVis, encoderNir, netG, startEpoch = loadModel(cfg, encoderNir, encoderVis, netG)
        optimizer = loadOptimizer(cfg, optimizer)
    else:
        startEpoch = 0

    # criterion
    l2Loss = torch.nn.MSELoss()
    l1Loss = torch.nn.L1Loss()
    smoothL1Loss = torch.nn.SmoothL1Loss()
    lossDict = {
        'l1': l1Loss,
        'l2': l2Loss,
        'smoothL1': smoothL1Loss
    }
    ipLoss = lossDict[cfg.G.TRAIN.IP_LOSS].cuda()
    pairLoss = lossDict[cfg.G.TRAIN.PAIR_LOSS].cuda()
    recLoss = lossDict[cfg.G.TRAIN.REC_LOSS].cuda()

    # dataloader
    trainLoader = torch.utils.data.DataLoader(
        GenDataset(imgRoot=cfg.G.DATASET.ROOT, protocolsRoot=cfg.G.DATASET.PROTOCOLS),
        batch_size=cfg.G.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=cfg.G.TRAIN.NUM_WORKERS, pin_memory=True, drop_last=False
    )

    # writer
    TIMESTAMP = "{0:%Y%m%dT%H%M%S}".format(datetime.now())
    writer = SummaryWriter(logdir=os.path.join(cfg.MISC.OUTPUT_PATH, 'run', '{}'.format(cfg.CFG_NAME)))

    for epoch in range(startEpoch, cfg.G.TRAIN.EPOCH):
        batchTime = AverageMeter()
        dataTime = AverageMeter()
        losses = AverageMeter()
        recLosses = AverageMeter()
        klLosses = AverageMeter()
        mmdLosses = AverageMeter()
        ipLosses = AverageMeter()
        pairLosses = AverageMeter()

        encoderVis.train()
        encoderNir.train()
        netG.train()
        netIP.eval()

        startTime = time.time()
        for i, batch in enumerate(trainLoader):
            dataTime.update(time.time() - startTime)

            imgNir = Variable(batch['0'].cuda())
            imgVis = Variable(batch['1'].cuda())

            img = torch.cat((imgNir, imgVis), dim=1)

            # encoder forward
            muNir, logvarNir = encoderNir(imgNir)
            muVis, logvarVis = encoderVis(imgVis)

            # re-parametrization
            zNir = reparameterize(muNir, logvarNir)
            zVis = reparameterize(muVis, logvarVis)

            # generator
            rec = netG(torch.cat((zNir, zVis), dim=1))

            # vae loss
            # lossRec = reconLoss(rec, img, True) / 2.
            lossRec = cfg.G.TRAIN.LAMBDA_REC * recLoss(rec, img) / 2.0
            lossKL = cfg.G.TRAIN.LAMBDA_KL * (klLoss(muNir, logvarNir).mean() + klLoss(muVis, logvarVis).mean()) / 2.0

            # mmd loss
            lossMMD = cfg.G.TRAIN.LAMBDA_MMD * torch.abs(zNir.mean(dim=0) - zVis.mean(dim=0)).mean()

            # identity preserving loss
            recNir = rec[:, 0:3, :, :]
            recVis = rec[:, 3:6, :, :]

            embedNir = F.normalize(netIP(rgb2gray(imgNir))[0], p=2, dim=1)
            embedVis = F.normalize(netIP(rgb2gray(imgVis))[0], p=2, dim=1)

            recEmbedNir = F.normalize(netIP(rgb2gray(recNir))[0], p=2, dim=1)
            recEmbedVis = F.normalize(netIP(rgb2gray(recVis))[0], p=2, dim=1)

            lossIP = cfg.G.TRAIN.LAMBDA_IP * (ipLoss(recEmbedNir, embedNir.detach()) + ipLoss(recEmbedVis, embedVis.detach())) / 2.0
            lossPair = cfg.G.TRAIN.LAMBDA_PAIR * pairLoss(recEmbedNir, recEmbedVis)

            if epoch < 2:
                loss = lossRec + 0.01 * lossKL + 0.01 * lossMMD + 0.01 * lossIP + 0.01 * lossPair
            else:
                loss = lossRec + lossKL + lossMMD + lossIP + lossPair
            losses.update(loss.item())
            recLosses.update(lossRec.item())
            klLosses.update(lossKL.item())
            mmdLosses.update(lossMMD.item())
            ipLosses.update(lossIP.item())
            pairLosses.update(lossPair.item())

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchTime.update(time.time() - startTime)
            startTime = time.time()

            scheduler.step(epoch)

            if i % cfg.G.TRAIN.PRINT_FREQ == 0:
                info = '==> Epoch: [{:0>4d}][{:3d}/{:3d}] Batch time: {:4.3f} Data time: {:4.3f} | '.format(
                    epoch, i, len(trainLoader), batchTime.avg, dataTime.avg
                )
                info += 'Loss: rec: {:4.3f} kl: {:4.3f} mmd: {:4.3f} ip: {:4.8f} pair: {:4.8f}'.format(
                    lossRec.item(), lossKL.item(), lossMMD.item(), lossIP.item(), lossPair.item()
                )
                print(info)

        # writer
        writer.add_scalar('loss/loss', losses.avg, epoch)
        writer.add_scalar('loss/recLoss', recLosses.avg, epoch)
        writer.add_scalar('loss/klLoss', klLosses.avg, epoch)
        writer.add_scalar('loss/mmdLoss', mmdLosses.avg, epoch)
        writer.add_scalar('loss/ipLoss', ipLosses.avg, epoch)
        writer.add_scalar('loss/pairLoss', pairLosses.avg, epoch)

        x = vutils.make_grid(imgNir.data, normalize=True, scale_each=True)
        writer.add_image('nir/imgNir', x, epoch)
        x = vutils.make_grid(imgVis.data, normalize=True, scale_each=True)
        writer.add_image('vis/imgVis', x, epoch)
        x = vutils.make_grid(recNir.data, normalize=True, scale_each=True)
        writer.add_image('nir/recNIR', x, epoch)
        x = vutils.make_grid(recVis.data, normalize=True, scale_each=True)
        writer.add_image('vis/recVis', x, epoch)

        noise = torch.zeros(cfg.G.TRAIN.BATCH_SIZE, cfg.G.TRAIN.HDIM).normal_(0, 1)
        noise = torch.cat((noise, noise), dim=1)
        noise = noise.cuda()
        fakeImg = netG(noise)
        x = vutils.make_grid(fakeImg[:, 0:3, :, :].data, normalize=True, scale_each=True)
        writer.add_image('fake/fakeNir', x, epoch)
        x = vutils.make_grid(fakeImg[:, 3:6, :, :].data, normalize=True, scale_each=True)
        writer.add_image('fake/fakeVis', x, epoch)

        # evaluation
        if not os.path.isdir(cfg.G.TEST.IMG_DUMP):
            os.makedirs(cfg.G.TEST.IMG_DUMP)
        if (epoch + 0) % cfg.G.TEST.FREQ == 0:
            noise = torch.zeros(cfg.G.TRAIN.BATCH_SIZE, cfg.G.TRAIN.HDIM).normal_(0, 1)
            noise = torch.cat((noise, noise), dim=1)
            noise = noise.cuda()

            fakeImg = netG(noise)

            vutils.save_image(fakeImg[:, 0:3, :, :].data, os.path.join(cfg.G.TEST.IMG_DUMP, '{}_epoch_{:03d}_fake_nir.png'.format(cfg.CFG_NAME, epoch)))
            vutils.save_image(fakeImg[:, 3:6, :, :].data, os.path.join(cfg.G.TEST.IMG_DUMP, '{}_epoch_{:03d}_fake_vis.png'.format(cfg.CFG_NAME, epoch)))
            vutils.save_image(imgNir.data, os.path.join(cfg.G.TEST.IMG_DUMP, '{}_epoch_{:03d}_img_nir.png'.format(cfg.CFG_NAME, epoch)))
            vutils.save_image(imgVis.data, os.path.join(cfg.G.TEST.IMG_DUMP, '{}_epoch_{:03d}_img_vis.png'.format(cfg.CFG_NAME, epoch)))
            vutils.save_image(recNir.data, os.path.join(cfg.G.TEST.IMG_DUMP, '{}_epoch_{:03d}_rec_nir.png'.format(cfg.CFG_NAME, epoch)))
            vutils.save_image(recVis.data, os.path.join(cfg.G.TEST.IMG_DUMP, '{}_epoch_{:03d}_rec_vis.png'.format(cfg.CFG_NAME, epoch)))

        if (epoch + 0) % cfg.G.TRAIN.SAVE_EPOCH == 0:
            saveOptimizer(cfg, optimizer, epoch)
            saveModel(cfg, encoderVis, encoderNir, netG, epoch)


if __name__ == "__main__":
    main()
