# DNN-Printer
Python Scritpt which can be embedded into PyTorch model to print the model size.


## Usage

1 Firstly, call the command `pip3 install DNN_printer` in server.

2 Then, add `from DNN_printer import DNN_printer` in PyTorch script

3 Put `DNN_printer(net, (3, 32, 32),batch_size)` in your code.

**Notice:** `net` is the model variance;`(3, 32, 32)` is the size of input data;`batch_size` is the number of batch size.

```
from DNN_printer import DNN_printer

batch_size = 512
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    // put the code here and you can get the result
    DNN_printer(net, (3, 32, 32),batch_size)
    ...
    ...
```

## result
```
Epoch: 0
------------------------------Happy every day! :)---------------------------------
-----------------------------Author: Peiyi & Ping---------------------------------
        Layer (type)         Output Shape      O-Size(MB)       Param #      P-Size(MB)
==================================================================================
            Conv2d-1      [64, 64, 32, 32]    16.0 MB         1,728 0.006591796875 MB
       BatchNorm2d-2      [64, 64, 32, 32]    16.0 MB           128 0.00048828125 MB
            Conv2d-3      [64, 64, 32, 32]    16.0 MB        36,864     0.140625 MB
       BatchNorm2d-4      [64, 64, 32, 32]    16.0 MB           128 0.00048828125 MB
            Conv2d-5      [64, 64, 32, 32]    16.0 MB        36,864     0.140625 MB
       BatchNorm2d-6      [64, 64, 32, 32]    16.0 MB           128 0.00048828125 MB
            Conv2d-7      [64, 64, 32, 32]    16.0 MB        36,864     0.140625 MB
       BatchNorm2d-8      [64, 64, 32, 32]    16.0 MB           128 0.00048828125 MB
            Conv2d-9      [64, 64, 32, 32]    16.0 MB        36,864     0.140625 MB
      BatchNorm2d-10      [64, 64, 32, 32]    16.0 MB           128 0.00048828125 MB
           Conv2d-11     [64, 128, 16, 16]     8.0 MB        73,728      0.28125 MB
      BatchNorm2d-12     [64, 128, 16, 16]     8.0 MB           256 0.0009765625 MB
           Conv2d-13     [64, 128, 16, 16]     8.0 MB       147,456       0.5625 MB
      BatchNorm2d-14     [64, 128, 16, 16]     8.0 MB           256 0.0009765625 MB
           Conv2d-15     [64, 128, 16, 16]     8.0 MB         8,192      0.03125 MB
      BatchNorm2d-16     [64, 128, 16, 16]     8.0 MB           256 0.0009765625 MB
           Conv2d-17     [64, 128, 16, 16]     8.0 MB       147,456       0.5625 MB
      BatchNorm2d-18     [64, 128, 16, 16]     8.0 MB           256 0.0009765625 MB
           Conv2d-19     [64, 128, 16, 16]     8.0 MB       147,456       0.5625 MB
      BatchNorm2d-20     [64, 128, 16, 16]     8.0 MB           256 0.0009765625 MB
           Conv2d-21       [64, 256, 8, 8]     4.0 MB       294,912        1.125 MB
      BatchNorm2d-22       [64, 256, 8, 8]     4.0 MB           512  0.001953125 MB
           Conv2d-23       [64, 256, 8, 8]     4.0 MB       589,824         2.25 MB
      BatchNorm2d-24       [64, 256, 8, 8]     4.0 MB           512  0.001953125 MB
           Conv2d-25       [64, 256, 8, 8]     4.0 MB        32,768        0.125 MB
      BatchNorm2d-26       [64, 256, 8, 8]     4.0 MB           512  0.001953125 MB
           Conv2d-27       [64, 256, 8, 8]     4.0 MB       589,824         2.25 MB
      BatchNorm2d-28       [64, 256, 8, 8]     4.0 MB           512  0.001953125 MB
           Conv2d-29       [64, 256, 8, 8]     4.0 MB       589,824         2.25 MB
      BatchNorm2d-30       [64, 256, 8, 8]     4.0 MB           512  0.001953125 MB
           Conv2d-31       [64, 512, 4, 4]     2.0 MB     1,179,648          4.5 MB
      BatchNorm2d-32       [64, 512, 4, 4]     2.0 MB         1,024   0.00390625 MB
           Conv2d-33       [64, 512, 4, 4]     2.0 MB     2,359,296          9.0 MB
      BatchNorm2d-34       [64, 512, 4, 4]     2.0 MB         1,024   0.00390625 MB
           Conv2d-35       [64, 512, 4, 4]     2.0 MB       131,072          0.5 MB
      BatchNorm2d-36       [64, 512, 4, 4]     2.0 MB         1,024   0.00390625 MB
           Conv2d-37       [64, 512, 4, 4]     2.0 MB     2,359,296          9.0 MB
      BatchNorm2d-38       [64, 512, 4, 4]     2.0 MB         1,024   0.00390625 MB
           Conv2d-39       [64, 512, 4, 4]     2.0 MB     2,359,296          9.0 MB
      BatchNorm2d-40       [64, 512, 4, 4]     2.0 MB         1,024   0.00390625 MB
           Linear-41              [64, 10] 0.00244140625 MB         5,130 0.01956939697265625 MB
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 600.00
Params size (MB): 42.63
Estimated Total Size (MB): 643.38
----------------------------------------------------------------

```


## Contributors:

Peiyi Hong & Ping
