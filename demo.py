import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils import data
from torch.autograd import Variable

from fcn import VGGNet,FCNs
from utils import get_color_index
from config import n_class


def img_loader(demo_path):
    imgs_path_list = []
    for looproot, _, filesnames in os.walk(demo_path):
        for filesname in filesnames:
            imgs_path_list.append(os.path.join(looproot, filesname.strip()))

    return imgs_path_list


def index2color(temp):
    '''
    根据class 分配颜色
    :param temp: temp.shape = [batch size, H, W]
    :return:
    '''
    # index与label颜色的调色板，模型输出是某个class的index(size <= class_num)，使用这个调色板转成相应颜色的张量
    colors_index = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    temp = np.array(temp,dtype=np.int)
    r = np.zeros_like(temp, dtype=np.int)
    g = np.zeros_like(temp, dtype=np.int)
    b = np.zeros_like(temp, dtype=np.int)
    for l in range(n_class):
        r[temp == l] = colors_index[l][0]
        g[temp == l] = colors_index[l][1]
        b[temp == l] = colors_index[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb


def val():
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = {}".format(device))
    # load data

    demo_path = os.path.join("demo")
    if not os.path.isdir(demo_path):
        os.mkdir(demo_path)

    imgs_path_list = img_loader(demo_path)

    # set model
    pretrain_model = VGGNet(requires_grad=True, remove_fc=True)
    model = FCNs(pretrained_net=pretrain_model, n_class=n_class)
    print("model loading success...")

    # set model running devices
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print("usable gpu num: {}".format(torch.cuda.device_count()))

    # load checkpoints
    load_ckpt_path = os.path.join("checkpoints",load_ckpt_name)

    if torch.cuda.is_available():
        checkpoint = torch.load(load_ckpt_path)
    else:
        checkpoint = torch.load(load_ckpt_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state'])
    last_best_iou = checkpoint['best_iou']
    start_epoch = checkpoint['epoch']

    print('Checkpoint resume success... last iou:{:.4%}'.format(last_best_iou))
    time_s = time.time()
    model.eval()
    with torch.no_grad():

        images = []
        for i, img_path in zip(range(len(imgs_path_list)), imgs_path_list):
            image = plt.imread(img_path)

            image = image[:,:,::-1]  # RGB => BGR
            images.append(image)

        images = np.array(images,dtype=np.float32)
        images = images.transpose([0,3,1,2])
        images_tensor = torch.tensor(images, dtype=torch.float32)
        images_tensor.to(device)
        outputs_val = model(images_tensor)

        for i in range(len(imgs_path_list)):
            plt.figure(i)
            plt.subplot(2,2,1)
            plt.title("Origin image")
            plt.imshow(images[i].transpose([1,2,0]))

            pred = outputs_val.data.max(1)[1].cpu().numpy()
            rgb_img = index2color(pred[i, :, :])

            plt.subplot(2,2,2)
            plt.title("Semantic Segmentation Predict, mIoU:{}".format(last_best_iou))
            plt.imshow(rgb_img.astype(np.int))

            plt.subplot(2,2,3)
            # show color2class bar
            range_cmap = [[i for i in range(n_class)]]
            # 自定义colormap
            c_map = mpl.colors.LinearSegmentedColormap.from_list('cmap', np.array(get_color_index()[:n_class]) / 255., 256)
            plt.imshow(range_cmap, cmap=c_map)
            plt.xticks(range_cmap[0],
                       ['Void', 'Road', 'Construction', 'Traffic light', 'Nature', 'Sky', 'Person', 'Vehicle'],
                       rotation=50)
        print("time used per image:{}".format(time_s/len(imgs_path_list)))
        plt.show()

if __name__ == "__main__":
    load_ckpt_name = "eph_13_iou_35.30%.ckpt.pth"
    val()