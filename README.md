# Semantic Segmentation Task, implemented by pytorch, with CityScapes dataset
## Architecture
* FCN with VGG (pretrain model from pytorch)
* optimizer : option:['Adam','rmrprop','SGD']
* criterion : torch.nn.BCEWithLogitsLoss
* hyper params: reference config.py
* about the class num reduce part, please reference label_changer.py, where show how some label class mix to one class(such as car,bike,truck to vehicle)
* if need to change the class num distribution, please rewrite code in cityscape_loader.py(Cityscapes_Loader. __init__ ,transform_classes)
## How to Run
* requirement list   
 CUDA    
 scipy = 0.19.1    
 pytorch  
 matplotlib  
 tensorboardX  
 PIL  
* change params in config.py (including the CityScapes datasets path)
* train with GPU(recommend)  
single GPU run `CUDA_VISIBLE_DEVICES=0 python train.py`
* run demo for your images   
move your images to path ./demo/   
run `python demo.py`  
## About params
* batch_size not too big, or else lead to GPU out of memory.
* the input images need to resize into (512,1024), or else cause out of memory.
* learn rate = 1 e-4 ,by using Adam
* 
## Result
* Result image predict
![score1.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/blob/master/score/score1.png)  
![score2.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/blob/master/score/score2.png)
![score3.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/blob/master/score/score3.png)
* The fig4 is the trend of Loss and mIoU within 100 epochs. We can see that after 47epochs the train_loss still decrease but val_loss begin increase, this would lead to overfitting, need to early stopping in this epoch.
![loss1.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/blob/master/score/loss1.png)
## Conclusion
* The different class group may result in different accuracy, for example, person class and sky class fuse together may lead to stop reducing of Loss, and mIoU will stay in a level no matter how epochs train.Just like the fig5 show.  
![stoploss1.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/blob/master/score/stoploss1.png)
* Use Adam optimizer could significant increase regression speed. As fig6 show.
![adamoptimizer1.png](https://github.com/hikaruzzz/instance-semantic-segmentation-pytorch/blob/master/score/adamoptimizer1.png)

