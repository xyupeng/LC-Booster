### Train
```
python train_ww_ws.py configs/cifar10/sym90_ww_ws.py
```
We do observe some variance on results of high noise ratio (e.g., cifar10 sym90), as in other noisy-label methods (AugDesc). 
We find the variance is largely determined by the first 100 epochs. 
So you can run several first 100 epochs with different seeds and resume from the best one to avoid full training many times.