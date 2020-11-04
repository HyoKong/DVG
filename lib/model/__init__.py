import torch
from model.generator import Encoder, Decoder
from model.lightCNN import LightCNN_29Layers_v2

def defineG(hdim=256):
    encoderNir = Encoder(hdim=hdim)
    encoderVis = Encoder(hdim)
    netG = Decoder(hdim=2*hdim)

    encoderNir = torch.nn.DataParallel(encoderNir).cuda()
    encoderVis = torch.nn.DataParallel(encoderVis).cuda()
    netG = torch.nn.DataParallel(netG).cuda()

    return encoderNir, encoderVis, netG

def defineIP(isTrain=False):
    netIP = LightCNN_29Layers_v2(training=isTrain)
    netIP = torch.nn.DataParallel(netIP).cuda()
    return netIP
