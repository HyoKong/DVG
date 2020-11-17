import os
import glob2


def main():
    protocolsPath = '/media/HDD8TB/hanyang/face/CASIA_VIS_NIR/protocols/'
    datasetRoot = '/media/HDD8TB/hanyang/face/CASIA_VIS_NIR/'

    gallery_file_list = 'vis_gallery_*.txt'
    probe_file_list = 'nir_probe_*.txt'
    # gallery_file_list = glob2.glob(args.root_path + '/' + args.protocols + '/' + gallery_file_list)
    # probe_file_list = glob2.glob(args.root_path + '/' + args.protocols + '/' + probe_file_list)
    gallery_file_list = glob2.glob(os.path.join(protocolsPath, gallery_file_list))
    probe_file_list = glob2.glob(os.path.join(protocolsPath, probe_file_list))
    gallery_file_list = sorted(gallery_file_list)[0:-1]
    probe_file_list = sorted(probe_file_list)[0:-1]

    idDict = {}  # {oldIdx: newIdx}
    newIdx = 0

    with open('generateList1.txt', 'w') as f:
        galleryList = []
        for root in gallery_file_list:
            with open(root, 'r') as f1:
                imgList = f1.readlines()
                imgList = [x.strip() for x in imgList]
                for line in imgList:
                    imgName = revise_name(line)
                    galleryList.append('VIS_Aligned/{}/{}'.format(imgName.split('.')[0], imgName))
            f1.close()
        probeList = []
        # excludeList = []
        for root in probe_file_list:
            with open(root, 'r') as f1:
                imgList = f1.readlines()
                imgList = [x.strip() for x in imgList]
                for line in imgList:
                    imgName = revise_name(line)
                    probe = 'NIR_Aligned/{}/{}'.format(imgName.split('.')[0], imgName)
                    if os.path.exists(os.path.join(datasetRoot, probe)):
                        probeList.append(probe)
            f1.close()
        # exclude probe path that do not exists in the aligned probe dataset.
        # excludePath = []
        # for root in probeList:
        #     if not os.path.exists(os.path.join(datasetRoot, root)):
        #         excludePath.append(root)

        # probeList = list(set(probeList) - set(excludePath))
        for root in probeList:
            _, newIdx, idDict = writeList(root, newIdx, idDict)
            f.write(writeList(root, newIdx, idDict)[0])
            f.write('\n')
        for root in galleryList:
            _, newIdx, idDict = writeList(root, newIdx, idDict)
            f.write(writeList(root, newIdx, idDict)[0])
            f.write('\n')
    f.close()


def writeList(imgName, newIdx, idDict):
    _, modality, idx, _ = imgName.split('_')[-4:]
    if modality == 'VIS':
        domain = 1
    else:
        domain = 0
    # idx = int(str(idx))
    if int(idx) not in idDict.keys():
        idDict[int(idx)] = newIdx
        newIdx += 1

    return '{} {} {}'.format(imgName, idDict[int(idx)], domain), newIdx, idDict


def revise_name(img_name):
    '''
    's2\\NIR\\10117\\016.bmp ==> s2_NIR_10117_016.bmp'
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


if __name__ == "__main__":
    main()
