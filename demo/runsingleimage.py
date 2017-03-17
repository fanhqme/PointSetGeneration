import cv2
import time
import numpy as np
import cPickle as pickle
import tensorflow as tf
import tflearn
import sys

BATCH_SIZE=1
HEIGHT=192
WIDTH=256

def loadModel(weightsfile):
	with tf.device('/cpu'):
		img_inp=tf.placeholder(tf.float32,shape=(BATCH_SIZE,HEIGHT,WIDTH,4),name='img_inp')
		x=img_inp
		#192 256
		x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x0=x
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		#96 128
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x1=x
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		#48 64
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x2=x
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		#24 32
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x3=x
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		#12 16
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x4=x
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		#6 8
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x5=x
		x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		x_additional=tflearn.layers.core.fully_connected(x,2048,activation='relu',weight_decay=1e-3,regularizer='L2')
		x_additional=tflearn.layers.core.fully_connected(x_additional,1024,activation='relu',weight_decay=1e-3,regularizer='L2')
		x_additional=tflearn.layers.core.fully_connected(x_additional,256*3,activation='linear',weight_decay=1e-3,regularizer='L2')
		x_additional=tf.reshape(x_additional,(BATCH_SIZE,256,3))
		x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
		x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x5))
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
		x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x4))
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x3))
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,3,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.reshape(x,(BATCH_SIZE,32*24,3))
		x=tf.concat([x_additional,x],axis=1)
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

def run_image(model,img_in,img_mask):
	(sess,img_inp,x)=model
	img_in=img_in*(1-img_mask[:,:,None])+191*img_mask[:,:,None]
	img_packed=np.dstack([img_in.astype('float32')/255,img_mask[:,:,None]])
	assert img_packed.shape==(HEIGHT,WIDTH,4)

	(ret,),=sess.run([x],feed_dict={img_inp:img_packed[None,:,:,:]})
	return ret

if __name__=='__main__':
	model=loadModel(sys.argv[3])
	img_in=cv2.imread(sys.argv[1])
	img_mask=cv2.imread(sys.argv[2],0)!=0
	fout=open(sys.argv[1]+'.txt','w')
	ret=run_image(model,img_in,img_mask)
	for x,y,z in ret:
		print >>fout,x,y,z
