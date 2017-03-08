## Using the code
Only training code fragments are uploaded. Data, pretrained model and more scripts are on the road.

Make sure you have python-numpy, python-opencv, tensorflow, CUDA, etc.
Some paths are configured in makefile. Overwrite them properly.

Compiling CUDA code
```
$ make
```

Usage of training script:

* Predict on validation set
```	
$ python train_nn.py [data=<path>] [dump=<path>] [num=<number>] predict
example: $ python train_nn.py data=data dump=dump num=3 predict
```
		
* Visualualize dumpped prediction (press space to view the next one)
```
$ python python visualizeptexample.v.py <path>/train_nn.v.pkl
example: $ python visualizeptexample.v.py dump/train_nn.v.pkl
```
		
* Train
```
$ python train_nn.py [data=<path>] [dump=<path>] train
example: $ python train_nn.py data=data dump=dump train
```

## Format of training data
.bin.gz files here are not gzipped file (sorry).
```
python traindataviewer.py data/0/0.gz
```
This shows a batch of training data. The loadBinFile function returns a tuple containing the color image, depth image, ground truth point cloud and model key names.