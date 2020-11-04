import os
import random
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class GenDataset(data.Dataset):
    def __init__(self, imgRoot, listFile):
        super().__init__()
        self.imgRoot = imgRoot
        self.listFile = listFile

        self.transform = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.ToTensor()
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

    def getPair(self, label, domainFlag):
        imgName = random.choice(self.pairDict[label][domainFlag])
        return imgName

    def fileReader(self):
        with open(self.listFile, 'r') as f:
            imgList = f.readlines()
            imgList = [x.strip() for x in imgList]
        pairDict = defaultdict({'0': [], '1':[]})
        imgDomain0List = []
        for line in imgList:
            imgName, label, domainFlag = line.strip().split(' ')
            pairDict[label][domainFlag].append(imgName)

            if domainFlag == '0':
                imgDomain0List.append(line)
        return imgDomain0List, pairDict