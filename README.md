## Using the code

Training scripts and a couple of trained demo networks are included. More demos and the complete set of data are on the road.

**KMZ file of the ShapeNet shapes used in this experiment are released! See the end of this page**
**The full training batches used in this experiment are released! See the end of this page**

Make sure you have python-numpy, python-opencv, tensorflow, tflearn, CUDA, etc.
Some paths are configured in makefile. Overwrite them properly.

### Running the demo

If you just want to try the demo, cd into the demo directory, and run
```
$ python runsingleimage.py 1.png 1_m.png twobranch_v1.pkl
$ python view.py 1.png.txt
```
The .pkl files can be found in the google drive:
- version 1 (used in the paper) https://drive.google.com/file/d/0B0gQFbJEIJ4kT3lCUy1UQVlZQnc/view?usp=sharing
- version 2 (improved) https://drive.google.com/file/d/0B0gQFbJEIJ4kUTc2cGlDeDh6VGs/view?usp=sharing

The first script runs the code on the image 1.png with segmentation mask 1_m.png using neural network weights twobranch_v1.pkl. Another set of weights twobranch_v2.pkl seems more robust. The input images must be of size 256x192. The second script visualizes the predicted point cloud. Move your mouse over the window to rotate the point cloud.

**If you want to try the networks on your own captured image, see ImageCaptureGuide.pdf first.**

We have also included a trained network corresponding to the R2N2 paper's setting. you can download runr2n2_128_v1.pkl from
https://drive.google.com/file/d/0B0gQFbJEIJ4kQVdpeVBNb2RJTlk/view?usp=sharing
and run
```
$ python runr2n2_128.py r1.png runr2n2_128_v1.pkl
$ python view.py r1.png.txt
```

### Training

If you are interested in training a network, here are the instructions.

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
A few minibatches of processed training data is in the data/ folder.

.bin.gz files here are not gzipped file (sorry).
```
python traindataviewer.py data/0/0.gz
```
This shows a batch of training data. The loadBinFile function returns a tuple containing the color image, depth image, ground truth point cloud and model key names.

Below is the complete set of training data. Download them all into the data/ folder.
https://www.dropbox.com/sh/68kfpqut2y75etz/AABtIn2LUMALTnULSTUr5ZlUa?dl=0

Below is more data that might be useful. Notice: you must use https.

https://shapenet.cs.stanford.edu/media/sampledata_220k.tar
