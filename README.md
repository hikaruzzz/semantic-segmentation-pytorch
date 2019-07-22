# Semantic Segmentation Task, implemented by pytorch
## Architecture
* FCN with VGG (pretrain model from pytorch)
* optimizer : option:['rmrprop','SGD']
* criterion : torch.nn.BCEWithLogitsLoss
* hyper params: reference config.py
## How to Run
* requirement list
1 cuda
2 scipy  = 0.19.1
3 
* train with GPU(recomment)  
run `CUDA_VISIBLE_DEVICES=0 python train.py`
## About params
* batch_size not too big, or else lead to GPU out of memory.
* the input images need to resize into (512,1024), or else cause out of memory.
* learn rate,
## Result
* day 7/22/2019 record, the mIoU can't increase over 35%.
* ![score1.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/edit/master/score1.png)
* ![score2.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/edit/master/score2.png)
