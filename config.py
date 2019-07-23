n_classes = 8

# dataset root path
#path_root = r"C:\Users\PC\Desktop\CityScapes"
path_root = r"/home/downloads/CityScapes/CityScapes"


# data loader
batch_size = 4

# cuda device set
#vaild_device_num = torch.cuda.device_count()
vaild_device_num = 1  # no used

# hyper paras
epochs = 10
n_class = 8
learn_rate = 1.0e-4
momentum = 0.99
weight_decay = 0.0005
step_size = 50  # decay LR 0.5 every 30 epoch
gamma = 0.5

optimizer_name = "rmsprop"  # option：['rmsprop','sgd']
#criterion = torch.nn.BCEWithLogitsLoss()

is_load_checkpoints = False
load_ckpt_name = "eph_50_iou_58.65%.ckpt.pth"


