import os
import time
import torch
import numpy as np
from torch.utils import data
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from cityscape_loader import Cityscapes_Loader
from fcn import VGGNet,FCNs
from utils import calc_iou
from config import *


def train():
    torch.manual_seed(1280)
    torch.cuda.manual_seed(1280)
    np.random.seed(1280)

    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = {}".format(device))
    # load data
    data_loader_train = Cityscapes_Loader(path_root=path_root,split="train",n_classes=8)
    data_loader_val = Cityscapes_Loader(path_root=path_root,split="val",n_classes=8)

    train_loader = data.DataLoader(data_loader_train,batch_size=batch_size,num_workers=16)
    val_loader = data.DataLoader(data_loader_val,batch_size=batch_size,num_workers=16)  # val batch_size=1

    # set model
    assert torch.cuda.is_available(), "先把下面的cuda()删掉，debug阶段，不支持cpu"
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.benchmark = True
    pretrain_model = VGGNet(requires_grad=True, remove_fc=True)
    model = FCNs(pretrained_net=pretrain_model, n_class=n_class)

    print("model loading success...")

    # set model running devices
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print("usable gpu num: {}".format(torch.cuda.device_count()))

    # set optimizer, lr_scheduler, loss function
    optimizer = None
    if optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(),lr=learn_rate,momentum=momentum,weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),lr=learn_rate,momentum=momentum,weight_decay=weight_decay)

    #criterion = mybasilLoss()
    criterion = torch.nn.BCEWithLogitsLoss()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # load checkpoints
    last_best_iou = -100.
    load_ckpt_path = os.path.join("checkpoints",load_ckpt_name)
    if is_load_checkpoints:
        if torch.cuda.is_available():
            checkpoint = torch.load(load_ckpt_path)
        else:
            checkpoint = torch.load(load_ckpt_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state'])
        last_best_iou = checkpoint['best_iou']
        start_epoch = checkpoint['epoch']

        print('Checkpoint resume success... last iou:{:.4%}'.format(last_best_iou))

    # train epoch
    best_iou = last_best_iou
    time_epoch = time.time()
    i = 0
    for epoch in range(epochs):
        time_step = time.time()
        for step, batch in enumerate(train_loader):
            lr_scheduler.step(epoch=epoch)  # 这个scheduler放在哪里？上一个for，还是这个for
            model.train()
            images = batch[0].to(device)
            #labels = batch[1].to(device)
            targets = batch[2].to(device)  # targets.shape=[batch, n_classes, H, W]

            optimizer.zero_grad()

            outputs = model(images)
            loss = None

            try:
                loss = criterion(input=outputs,target=targets)
            except:
                torch.cuda.empty_cache()
                loss = criterion(input=outputs, target=targets)

            loss.backward()
            optimizer.step()

            if(step%100 == 0):
                print("train after setp:{}, Loss:{:.4%}, time used:{:.4} s".format(step,loss.item(),time.time()-time_step))
                time_step = time.time()
                writer.add_scalars('ForTestOnly_record/Loss', {'train_loss': loss}, i + 1)
                i += 1

        # each epoch save the checkpoint
        model.eval()
        with torch.no_grad():
            total_iou = 0
            for step_val, batch_val in enumerate(val_loader):
                images_val = batch_val[0].to(device)
                labels_val = batch_val[1].to(device)
                targets_val = batch_val[2].to(device)

                outputs_val = model(images_val)
                loss_val = criterion(input=outputs_val,target=targets_val)

                pred = outputs_val.data.max(1)[1].cpu().numpy()  # 将mask格式[
                gt = labels_val.data.cpu().numpy()
                #break  # for only one val batch

                total_iou += calc_iou(pred,gt,n_class)

            mean_iou = total_iou/step_val
            print("epoch:{},loss_val:{:.4%},iou:{:.2%},total time used:{:.4}s".format(epoch + 1, loss_val, mean_iou,time.time()-time_epoch))
            writer.add_scalars('Train_record/Loss',{'train_loss':loss,'val_loss':loss_val}, epoch+1)
            writer.add_scalars('Train_record/iou',{"iou":mean_iou}, epoch+1)

            time_epoch = time.time()
            if mean_iou >= best_iou:
                best_iou = mean_iou
                state ={
                    "epoch":epoch + 1,
                    "model_state":model.state_dict(),
                    "best_iou":best_iou,
                }
                if not os.path.isdir('./checkpoints'):
                    os.mkdir('./checkpoints')
                save_path = os.path.join('./checkpoints', "eph_{}_iou_{:.2%}.ckpt.pth".format(epoch+1,best_iou))
                torch.save(state,save_path)
                print("checkpoint saved success")

    writer.close()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter()
    train()