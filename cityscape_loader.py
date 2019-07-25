import os
import numpy as np
import scipy.misc as scipym
import matplotlib.pyplot as plt
import torch
import warnings

import utils

from torch.utils import data


# scipy  = 0.19.1
warnings.filterwarnings("ignore")
# ignore the scipy.misc.pilutil.py warming: futurewarning: from float to np.floating


class Cityscapes_Loader(data.Dataset):
    '''
    重构了Dataset里面的方法，其中__getitem__()是返回单个图像的，交由DataLoader进行batch size打包
    '''

    def __init__(self,path_root,split,n_classes,is_transform=False,label_type="gtFine_labelIds.png"):
        '''
        只需要传入一个路径，通过__getitem__递归读取每张图
        :param path_root: x/CityScapes/
        :param split:   "test" or "train" or "val" in the dataset
        :param n_classes:  需要缩减至的classes数量
        :param is_transform:
        :param label_type: 使用labelids或instancelds.png
        '''
        self.root = path_root
        self.is_transform =is_transform
        self.n_classes = n_classes
        self.mean_rgb = [0.0,0.0,0.0]  # cityscapes 数据集的mean
        self.files_len = 0
        self.split = split

        self.H = 512  # origin image shape / 2
        self.W = 1024

        # leftimg8bit的目录为 root(包含/CityScapes) + leftimg8bit
        self.path_leftImg8bit_split = os.path.join(self.root,"leftImg8bit",self.split) #images_base
        self.path_gtFine_spilt = os.path.join(self.root,"gtFine",self.split) # annotations_base

        # files是整个leftImg8bit的目录树 [leftImg8bit--train/test/val--city_name--file_name]
        self.imgs_path_list = self.getFilesList(path_root)
        self.label_type = label_type
        assert self.__len__() > 0, "Error: not found image in path = {}".format(self.root)

        #合并class ，将一些类别合并成一个class（比如单车，汽车=》vehicle）,缩减classes数量，如现在是35 缩到 n个类
        transform_classes = utils.get_classes(n_classes)

        self.classes_map = dict(zip(range(-1, 35), transform_classes))

        # index与label颜色的调色板，模型输出是某个class的index(size <= class_num)，使用这个调色板转成相应颜色的张量
        self.colors_index = utils.get_color_index()

    def __len__(self):
        '''
        :return: 整个split中的所有图片len（包含各city）
        '''
        return len(self.imgs_path_list)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        :return img  img.shape = [3, H, W]
        :return  label.shape=(H,W) , target.shape=(n_classes, H, W)
        """
        img_path = self.imgs_path_list[index]
        # 根据leftimg8bit的文件列表，从gtFine中找到对应名称的label image，
        label_path = os.path.join(self.path_gtFine_spilt,
                                img_path.split(os.sep)[-2],  # 取倒数第二个（city）
                                img_path.split(os.sep)[-1][:-15] + self.label_type # 把leftImg..png换成gtFine..png
                                )

        # 读取 单个img和对应的 label 图
        img = scipym.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        assert img_path.split(os.sep)[-1][:15] == label_path.split(os.sep)[-1][:15], "!Error: not found image: {}".format(label_path)
        label = scipym.imread(label_path)
        label = self.reduceClasses(np.array(label, dtype=np.uint8))

        if self.is_transform:
            img, label = self.transform(img, label)

        # transform 方法
        img = scipym.imresize(img, (self.H, self.W))  # uint8 with RGB mode
        img = img[:,:,::-1]  # RGB  ->  BGR
        img = np.array(img,dtype=float)
        img -= self.mean_rgb
        img = img / 255.
        img = img.transpose([2,0,1])

        label = label.astype(np.float64)
        label = scipym.imresize(label, (self.H, self.W), "nearest", mode="F")
        label = label.astype(np.int32)

        target = np.zeros([self.n_classes, self.H, self.W],dtype=np.int32)
        for c in range(self.n_classes):
            target[c][label == c] = 1

        # new_target = np.zeros([self.H,self.W],dtype=np.int32)
        # for c in range(self.n_classes):
        #     new_target[target[c] == 1] = c
        #
        # plt.imshow(new_target)
        # plt.show()
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()
        target = torch.from_numpy(target).float()

        return img, label, target  # label.shape=(H,W) , target.shape=(n_classes, H, W)

    def transform(self,img,lbl):
        '''
        transform the images, such as crop, rotate
        :param img:
        :param lbl:
        :return:
        '''
        print("transform part is not done yet")

        return img,lbl

    def getFilesList(self,path_root):
        '''
        获取cityscapes/leftImg8bit/split/下面所有图片（把各个city的都结合到一起）
        :param path_root:
        :return:
        '''
        gtFine_files_path = os.path.join(path_root,"gtFine",self.split)
        leftImg8bit_files_path = os.path.join(path_root,"leftImg8bit",self.split)
        imgs_path_list = []
        for looproot, _, filesnames in os.walk(leftImg8bit_files_path):
            for filesname in filesnames:
                imgs_path_list.append(os.path.join(looproot, filesname.strip()))

        return imgs_path_list

    def index2color(self, temp):
        temp = temp.numpy()  # tensor to numpy
        r = np.zeros_like(temp, dtype=np.int)
        g = np.zeros_like(temp, dtype=np.int)
        b = np.zeros_like(temp, dtype=np.int)
        for l in range(self.n_classes):
            r[temp == l] = self.colors_index[l][0]
            g[temp == l] = self.colors_index[l][1]
            b[temp == l] = self.colors_index[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return rgb

    def reduceClasses(self,img):
        # 将img里面的pixel逐个转换成结合后的class index
        keys_list = list(self.classes_map.keys())
        for key in keys_list:
            img[img == key] = self.classes_map[key]

        return img


def forDebugOnly():
    path_root = r"C:\Users\PC\Desktop\CityScapes"

    split = "train"
    n_classes = 20
    c_loader = Cityscapes_Loader(path_root,split,n_classes,is_transform=False)
    print("list len:",c_loader.__len__())
    trainloader = data.DataLoader(c_loader, batch_size=1, num_workers=1)
    for i, data_samples in enumerate(trainloader):
        imgs, labels, target = data_samples
        plt.figure(i)
        plt.subplot(1,2,1)
        plt.imshow(np.array(np.transpose(imgs[0],[1,2,0])*255.,dtype=np.int))
        plt.subplot(1,2,2)
        plt.imshow(np.array(c_loader.index2color(labels[0]),dtype=np.int))
    plt.show()


if __name__ == "__main__":

    forDebugOnly()


