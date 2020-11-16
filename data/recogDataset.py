import os
import copy
import math
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


def defaultLoader(path):
    img = Image.open(path).convert('L')
    return img


def defaultListReader(fileListPath):
    '''

    :param fileListPath:
    :return:
    '''
    imgList = []
    with open(fileListPath, 'r') as f:
        for line in f.readlines():
            imgPath, label, domain = line.strip().split(' ')
            imgList.append((imgPath, int(label), int(domain)))
    f.close()
    return imgList


class ImageList(data.Dataset):
    def __init__(self, root, fileListPath):
        self.root = root
        self.imgList = defaultListReader(fileListPath)
        self.transform = transforms.Compose([
            transforms.RandomCrop(128),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        imgPath, label, domain = self.imgList[idx]
        img = Image.open(os.path.join(self.root, imgPath)).convert('L')
        img = self.transform(img)
        return {'img': img, 'label': label, 'domain': domain}

    def __len__(self):
        return len(self.imgList)


class SeparateBatchSampler(object):
    def __init__(self, realDataIdx, fakeDataIdx, batchSize=128, ratio=0.5, putBack=False):
        self.batchSize = batchSize
        self.ratio = ratio
        self.realDataNum = len(realDataIdx)
        self.fakeDataNum = len(fakeDataIdx)
        self.maxNumImage = max(self.realDataNum, self.fakeDataNum)

        self.realDataIdx = realDataIdx
        self.fakeDataIdx = fakeDataIdx

        self.processedIdx = copy.deepcopy(self.realDataIdx)

    def __len__(self):
        return self.maxNumImage // (int(self.batchSize * self.ratio))

    def __iter__(self):
        batchSizeRealData = int(math.floor(self.ratio * self.batchSize))
        batchSizeFakeData = self.batchSize - batchSizeRealData

        self.processedIdx = copy.deepcopy(self.realDataIdx)
        randRealDataIdx = np.random.permutation(len(self.realDataIdx) // 2)
        for i in range(self.__len__()):
            batch = []

            idxFakeData = random.sample(self.fakeDataIdx, batchSizeFakeData // 2)
            for j in range(batchSizeRealData // 2):
                idx = randRealDataIdx[i * batchSizeRealData + j % (self.realDataNum // 2)]
                batch.append(self.processedIdx[2 * idx])
                batch.append(self.processedIdx[2 * idx + 1])

            for idx in idxFakeData:
                batch.append(2 * idx + self.realDataNum)
                batch.append(2 * idx + 1 + self.realDataNum)
            yield batch


class SeparateImageList(data.Dataset):
    def __init__(self, realDataPath, realListPath, fakeDataPath, fakeTotalNum, ratio=0.5):
        '''

        :param realDataPath: Image Root
        :param realListPath:
        :param fakeDataPath:
        :param fakeTotalNum:
        :param ratio:
        '''
        self.transform = transforms.Compose([
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        # load real nir/vis data.
        realDataList, realDataIdx = self.listReader(realDataPath, realListPath)

        # load fake nir/vis data from noise
        idx = np.random.permutation(fakeTotalNum)
        fakeDataList = []  # generated images, one nir path, one vis path, iteratively.
        fakeDataIdx = []
        for i in range(0, fakeTotalNum):
            fakeImgName = str(idx[i] + 1) + '.jpg'
            fakeNirPath = os.path.join(fakeDataPath, 'nir_noise', fakeImgName)
            fakeVisPath = os.path.join(fakeDataPath, 'vis_noise', fakeImgName)
            fakeDataList.append((fakeNirPath, -1, 0))
            fakeDataList.append((fakeVisPath, -1, 1))
            fakeDataIdx.append(i)

        self.realDataIdx = realDataIdx
        self.fakeDataIdx = fakeDataIdx

        realDataList.append(fakeDataList)
        self.dataList = realDataList

        self.ratio = ratio

        print('Real: {} Fake: {} Total: {} Ratio: {}'.format(len(self.realDataIdx), len(self.fakeDataIdx), len(self.dataList), self.ratio))

    def __getitem__(self, idx):
        imgPath, label, domain = self.dataList[idx]
        img = Image.open(imgPath).convert('L')

        img = self.transform(img)
        return {'img': img, 'label': label, 'domain': domain}

    def __len__(self):
        return len(self.dataList)

    def listReader(self, rootPath, fileListPath):
        '''

        :param rootPath: root path of the images.
        :param fileListPath: txt file or real nir/vis dataset, 'imgName, label, domain'
        :return:
        '''
        imgList = []
        imgIdx = []
        idx = 0
        with open(fileListPath, 'r') as f:
            for line in f.readlines():
                imgName, label, domain = line.strip().split(' ')
                imgPath = os.path.join(rootPath, imgName)
                imgList.append((imgPath, int(label), int(domain)))
                imgIdx.append(idx)
                idx += 1
        f.close()
        return imgList, imgIdx

    def getIdx(self):
        return self.realDataIdx, self.fakeDataIdx
