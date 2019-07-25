import numpy as np
from config import *


# Calculates class intersections over unions
def iou(pred, gt):
    ious = []
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = gt == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious

def calc_iou(pred,gt,n_classes):
    mask = (gt >= 0) & (pred < n_classes)
    hist = np.bincount(n_classes * gt[mask].astype(int) + pred[mask], minlength=n_classes ** 2
                       ).reshape(n_classes, n_classes)

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)

    return mean_iou

def get_color_index():
    # 返回RGBcolor，按照class的顺序
    # 调色板
    colors_index = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [0,200,0],  # nature
        [0,191,255],  # sky
        [0,0,255], # human
        [220, 220, 0], # vehicle
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
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
    return colors_index

def get_classes(classes_num):
    '''
    合并class ，将一些类别合并成一个class（比如单车，汽车=》vehicle）,缩减classes数量，如现在是35 缩到 8个类
    详细label的class目录，可以参考 label_changer.py的表格
    :param classes_num: optional[2,8,20] represent (person&void),(8classes),(20classes) respectively
    :return:
    '''
    if classes_num == 2:
        # only person & void class
        transform_classes = np.zeros(35,dtype=np.int)
        transform_classes[1:25] = 0
        transform_classes[25:27] = 1
        transform_classes[27:35] = 0
        transform_classes[0] = 0

        return transform_classes
    elif classes_num == 8:
        transform_classes = np.zeros(35, dtype=np.int)
        transform_classes[1:8] = 0  # 0-7 -> 1-8 : void
        transform_classes[8:12] = 1  # 7-11 -> 8-12 :
        transform_classes[12:18] = 2
        transform_classes[18:22] = 3
        transform_classes[22:24] = 4
        transform_classes[24:25] = 5
        transform_classes[25:27] = 6
        transform_classes[27:35] = 7
        transform_classes[0] = 7  # 这个是train_idx = -1的那个class，归为7类，（占了transform_classes的第一位）后面idx 的都要往后退一位

        return transform_classes
    elif classes_num == 20:
        transform_classes = np.zeros(35, dtype=np.int)
        transform_classes[1:8] = 0  # 0-7 -> 1-8 : void
        transform_classes[8:9] = 1  # road
        transform_classes[9] = 2  # sidewalk
        transform_classes[10] = 0 # parking
        transform_classes[11] = 0  # rail track
        transform_classes[12] = 3  # building
        transform_classes[13] = 4  # wall
        transform_classes[14] = 5  # fence
        transform_classes[15:18] = 0  # bridge
        transform_classes[18] = 6  # pole
        transform_classes[19] = 0  # polegroup
        transform_classes[20] = 7  # traffic light
        transform_classes[21] = 8  # traffic sign
        transform_classes[22] = 9  # vegetation
        transform_classes[23] = 10  # terrain
        transform_classes[24] = 11  # sky
        transform_classes[25] = 12  # person
        transform_classes[26] = 13  # rider
        transform_classes[27] = 14  # car
        transform_classes[28] = 15  # truck
        transform_classes[29] = 16  # bus
        transform_classes[30:32] = 0  # caravan -> void
        transform_classes[32] = 17  # train
        transform_classes[33] = 18  # motorcycle
        transform_classes[34] = 18  # bicycle
        transform_classes[0] = 19  # license plate -> void

        return transform_classes

