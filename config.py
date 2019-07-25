
# dataset root path
#path_root = r"C:\Users\PC\Desktop\CityScapes"
path_root = r"/home/downloads/CityScapes/CityScapes"


# data loader
batch_size = 4

# cuda device set
#vaild_device_num = torch.cuda.device_count()
vaild_device_num = 1  # no used

# hyper paras
epochs = 100
n_class = 8
learn_rate = 1.0e-4
momentum = 0.9  # 利用了类似与移动指数加权平均的方法对参数进行平滑处理，减少波动幅度（解决SGD波动幅度大）
weight_decay = 0.0005
step_size = 100  # decay LR 0.5 every 30 epoch
gamma = 0.1

optimizer_name = "adam"  # option：['rmsprop','sgd','adam']
#criterion = torch.nn.BCEWithLogitsLoss()

is_load_checkpoints = False
load_ckpt_name = "eph_50_iou_58.65%.ckpt.pth"



