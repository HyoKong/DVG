import os
import random
import glob2
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class GenDataset(data.Dataset):
    def __init__(self, imgRoot, protocolsRoot):
        super().__init__()
        self.imgRoot = imgRoot
        self.protocolsRoot = protocolsRoot

        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ])

        self.imgDomain0List, self.pairDict = self.fileReader()

    def __getitem__(self, idx):
        imgLine = self.imgDomain0List[idx]
        imgNameDomain0, label, _ = imgLine.strip().split(' ')
        imgNameDomain1 = self.getPair(label, '1')
        imgDomain0 = Image.open(os.path.join(self.imgRoot, imgNameDomain0))
        imgDomain1 = Image.open(os.path.join(self.imgRoot, imgNameDomain1))

        imgDomain0 = self.transform(imgDomain0)
        imgDomain1 = self.transform(imgDomain1)
        return {'0': imgDomain0, '1': imgDomain1}

    # def getProtocolList(self):
    #     galleryFileList = 'vis_gallery_*.txt'
    #     probeFileList = 'nir_probe_*.txt'
    #     galleryFileList = glob2.glob(os.path.join(self.protocolsRoot, galleryFileList))
    #     probeFileList = glob2.glob(os.path.join(self.protocolsRoot, probeFileList))
    #     galleryFileList = sorted(galleryFileList)[0:-1]
    #     probeFileList = sorted(probeFileList)[0:-1]

    def __len__(self):
        flag = 1e8
        for k,v in self.pairDict.items():
            count = 0
            for label, imgList in v.items():
                count += len(imgList)
            if flag > count:
                flag = count
        return flag

    def getPair(self, label, domainFlag):
        imgName = random.choice(self.pairDict[domainFlag][label])
        return imgName

    def fileReader(self):
        with open(self.protocolsRoot, 'r') as f:
            imgList = f.readlines()
            imgList = [x.strip() for x in imgList]
        pairDict = {'0': {}, '1':{}}
        imgDomain0List = []
        for line in imgList:
            imgName, label, domainFlag = line.strip().split(' ')
            if label not in pairDict[domainFlag].keys():
                pairDict[domainFlag][label] = []
            pairDict[domainFlag][label].append(imgName)

            if domainFlag == '0':
                imgDomain0List.append(line)
        return imgDomain0List, pairDict