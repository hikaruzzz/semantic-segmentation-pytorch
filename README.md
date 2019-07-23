# Semantic Segmentation Task, implemented by pytorch
## Architecture
* FCN with VGG (pretrain model from pytorch)
* optimizer : option:['rmrprop','SGD']
* criterion : torch.nn.BCEWithLogitsLoss
* hyper params: reference config.py
* about the class num reduce part, please reference label_changer.py, where show how some label class mix to one class(such as car,bike,truck to vehicle)
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
* learn rate = 1 e-4 ,by using SGD
* 
## Result
* Result image predict, with mIoU:?? , val loss:??  
![score1.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/edit/master/score/score1.png)  
![score2.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/edit/master/score/score2.png)
* The trend of Loss and mIoU in 100 epochs  
![loss1.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/edit/master/score/loss1.png)
## Conclusion
* The different class group may result in different accuracy, for example, person class and sky class fuse together may lead to stop reducing of Loss, and mIoU will stay in a level no matter how epochs train.Just like the fig4 show.  
![stoploss1.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/edit/master/score/stoploss1.png)
