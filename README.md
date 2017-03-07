Only training code fragments are uploaded. More data and scripts on the road.

installation:
	$ make

Usage:

	Predict:
	
		$ python train_nn.py [data=<path>] [dump=<path>] [num=<number>] predict
		example: $ python train_nn.py data=data dump=dump num=3 predict
		
	Visualization:
	
		$ python python visualizeptexample.v.py <path>/train_nn.v.pkl
		example: $ python visualizeptexample.v.py dump/train_nn.v.pkl
		
	Train:
	
		$ python train_nn.py [data=<path>] [dump=<path>] train
		example: $ python train_nn.py data=data dump=dump train
