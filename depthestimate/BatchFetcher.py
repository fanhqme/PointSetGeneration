import sys
import numpy as np
import cv2
import random
import math
import os
import time
import zlib
import socket
import threading
import Queue
import sys
import cPickle as pickle
import show3d

FETCH_BATCH_SIZE=32
BATCH_SIZE=32
HEIGHT=192
WIDTH=256
POINTCLOUDSIZE=16384
OUTPUTPOINTS=1024
REEBSIZE=1024

class BatchFetcher(threading.Thread):
	def __init__(self, dataname):
		super(BatchFetcher,self).__init__()
		self.queue=Queue.Queue(64)
		self.stopped=False
		self.datadir = dataname
		self.bno=0
	def work(self,bno):
		path = os.path.join(self.datadir,'%d/%d.gz'%(bno//1000,bno))
		if not os.path.exists(path):
			self.stopped=True
			print "error! data file not exists: %s"%path
			print "please KILL THIS PROGRAM otherwise it will bear undefined behaviors"
			assert False,"data file not exists: %s"%path
		binfile=zlib.decompress(open(path,'r').read())
		p=0
		color=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*HEIGHT*WIDTH*3],dtype='uint8').reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH,3))
		p+=FETCH_BATCH_SIZE*HEIGHT*WIDTH*3
		depth=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*HEIGHT*WIDTH*2],dtype='uint16').reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH))
		p+=FETCH_BATCH_SIZE*HEIGHT*WIDTH*2
		rotmat=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*3*3*4],dtype='float32').reshape((FETCH_BATCH_SIZE,3,3))
		p+=FETCH_BATCH_SIZE*3*3*4
		ptcloud=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*POINTCLOUDSIZE*3],dtype='uint8').reshape((FETCH_BATCH_SIZE,POINTCLOUDSIZE,3))
		ptcloud=ptcloud.astype('float32')/255
		beta=math.pi/180*20
		viewmat=np.array([[
			np.cos(beta),0,-np.sin(beta)],[
			0,1,0],[
			np.sin(beta),0,np.cos(beta)]],dtype='float32')
		rotmat=rotmat.dot(np.linalg.inv(viewmat))
		for i in xrange(FETCH_BATCH_SIZE):
			ptcloud[i]=((ptcloud[i]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
		p+=FETCH_BATCH_SIZE*POINTCLOUDSIZE*3
		reeb=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*REEBSIZE*2*4],dtype='uint16').reshape((FETCH_BATCH_SIZE,REEBSIZE,4))
		p+=FETCH_BATCH_SIZE*REEBSIZE*2*4
		keynames=binfile[p:].split('\n')
		reeb=reeb.astype('float32')/65535
		for i in xrange(FETCH_BATCH_SIZE):
			reeb[i,:,:3]=((reeb[i,:,:3]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
		data=np.zeros((FETCH_BATCH_SIZE,HEIGHT,WIDTH,4),dtype='float32')
		data[:,:,:,:3]=color*(1/255.0)
		data[:,:,:,3]=depth==0
		validating=np.array([i[0]=='f' for i in keynames],dtype='float32')
		return (data,ptcloud,validating)
	def run(self):
		while self.bno<300000 and not self.stopped:
			self.queue.put(self.work(self.bno%300000))
			self.bno+=1
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	def shutdown(self):
		self.stopped=True
		while not self.queue.empty():
			self.queue.get()

if __name__=='__main__':
	dataname = "YTTRBtraindump_220k"
	fetchworker = BatchFetcher(dataname)
	fetchworker.bno=0
	fetchworker.start()
	for cnt in xrange(100):
		data,ptcloud,validating = fetchworker.fetch()
		validating = validating[0]!=0
		assert len(data)==FETCH_BATCH_SIZE
		for i in range(len(data)):
			cv2.imshow('data',data[i])
			while True:
				cmd=show3d.showpoints(ptcloud[i])
				if cmd==ord(' '):
					break
				elif cmd==ord('q'):
					break
			if cmd==ord('q'):
				break


