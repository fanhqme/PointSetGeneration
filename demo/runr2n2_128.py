import cv2
import time
import numpy as np
import cPickle as pickle
import tensorflow as tf
import tflearn
import sys

BATCH_SIZE=1
HEIGHT=128
WIDTH=128

def loadModel(weightsfile):
	with tf.device('/cpu'):
		img_inp=tf.placeholder(tf.float32,shape=(BATCH_SIZE,HEIGHT,WIDTH,3),name='img_inp')
		x=img_inp
#128 128
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x1=x
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#64 64
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x2=x
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#32 32
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x3=x
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#16 16
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x4=x
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#8 8
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x5=x
		x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		x_additional=tflearn.layers.core.fully_connected(x,2048,activation='relu',weight_decay=1e-3,regularizer='L2')
		x_additional=tflearn.layers.core.fully_connected(x_additional,1024,activation='relu',weight_decay=1e-3,regularizer='L2')
		x_additional=tflearn.layers.core.fully_connected(x_additional,256*3,activation='linear',weight_decay=1e-3,regularizer='L2')
		x_additional=tf.reshape(x_additional,(BATCH_SIZE,256,3))
		#x=tflearn.layers.core.fully_connected(x,3072,activation='relu',weight_decay=1e-3,regularizer='L2')
		#x=tf.reshape(x,(BATCH_SIZE,3,4,256))
		x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[8,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
		x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x5))
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[16,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
		x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x4))
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,3*3,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.reshape(x,(BATCH_SIZE,16*16*3,3))
		x=tf.concat([x_additional,x],1)
		x=tf.reshape(x,(BATCH_SIZE,1024,3))
	sess=tf.Session('')
	sess.run(tf.global_variables_initializer())
	loaddict={}
	fin=open(weightsfile,'rb')
	while True:
		try:
			v,p=pickle.load(fin)
		except EOFError:
			break
		loaddict[v]=p
	fin.close()
	for t in tf.trainable_variables():
		if t.name not in loaddict:
			print 'missing',t.name
		else:
			sess.run(t.assign(loaddict[t.name]))
			del loaddict[t.name]
	for k in loaddict.iteritems():
		if k[0]!='Variable:0':
			print 'unused',k
	return (sess,img_inp,x)

def run_image(model,img_in):
	(sess,img_inp,x)=model
	assert img_in.shape==(HEIGHT,WIDTH,3)
	img_in=np.float32(img_in)/255.0
	(ret,),=sess.run([x],feed_dict={img_inp:img_in[None,:,:,:]})
	return ret

if __name__=='__main__':
	model=loadModel(sys.argv[2])
	img_in=cv2.imread(sys.argv[1])
	fout=open(sys.argv[1]+'.txt','w')
	ret=run_image(model,img_in)
	for x,y,z in ret:
		print >>fout,x,y,z
