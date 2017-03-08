import numpy as np
import cv2
import zlib
import math

BATCH_SIZE=32
HEIGHT=192
WIDTH=256
POINTCLOUDSIZE=16384
OUTPUTPOINTS=1024
REEBSIZE=1024

def loadBinFile(path):
	binfile=zlib.decompress(open(path,'rb').read())
	p=0
	color=np.fromstring(binfile[p:p+BATCH_SIZE*HEIGHT*WIDTH*3],dtype='uint8').reshape((BATCH_SIZE,HEIGHT,WIDTH,3))
	p+=BATCH_SIZE*HEIGHT*WIDTH*3
	depth=np.fromstring(binfile[p:p+BATCH_SIZE*HEIGHT*WIDTH*2],dtype='uint16').reshape((BATCH_SIZE,HEIGHT,WIDTH))
	p+=BATCH_SIZE*HEIGHT*WIDTH*2
	rotmat=np.fromstring(binfile[p:p+BATCH_SIZE*3*3*4],dtype='float32').reshape((BATCH_SIZE,3,3))
	p+=BATCH_SIZE*3*3*4
	ptcloud=np.fromstring(binfile[p:p+BATCH_SIZE*POINTCLOUDSIZE*3],dtype='uint8').reshape((BATCH_SIZE,POINTCLOUDSIZE,3))
	ptcloud=ptcloud.astype('float32')/255
	beta=math.pi/180*20
	viewmat=np.array([[
		np.cos(beta),0,-np.sin(beta)],[
		0,1,0],[
		np.sin(beta),0,np.cos(beta)]],dtype='float32')
	rotmat=rotmat.dot(np.linalg.inv(viewmat))
	for i in xrange(BATCH_SIZE):
		ptcloud[i]=((ptcloud[i]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
	p+=BATCH_SIZE*POINTCLOUDSIZE*3
	some_other_thing=np.fromstring(binfile[p:p+BATCH_SIZE*REEBSIZE*2*4],dtype='uint16').reshape((BATCH_SIZE,REEBSIZE,4))
	p+=BATCH_SIZE*REEBSIZE*2*4
	keynames=binfile[p:].split('\n')
	data=np.zeros((BATCH_SIZE,HEIGHT,WIDTH,4),dtype='float32')
	data[:,:,:,:3]=color*(1/255.0)
	data[:,:,:,3]=depth==0
	validating=np.array([i[0]=='f' for i in keynames],dtype='float32')
	return color,depth,ptcloud,keynames

if __name__=='__main__':
	def plotimggrid(imgs,bgvalue=0,hpadding=0,vpadding=0):
		if len(imgs)==0:
			return np.zeros((1,1,3),dtype='uint8')^bgvalue
		ih=max([i.shape[0] for i in imgs])
		iw=max([i.shape[1] for i in imgs])
		w=min(len(imgs),max(1,int(np.ceil((len(imgs)*ih*iw)**0.5/iw))))
		h=((len(imgs)+(w-1))//w)
		output=np.zeros((h*ih+(h-1)*vpadding,w*iw+(w-1)*hpadding,3),dtype='uint8')^bgvalue
		for i in xrange(len(imgs)):
			x0=(i//w)*(ih+vpadding)
			y0=(i%w)*(iw+hpadding)
			output[x0:x0+imgs[i].shape[0],y0:y0+imgs[i].shape[1]]=imgs[i]
		return output
	import sys
	import show3d
	if len(sys.argv)<2:
		print 'python traindataviewer.py data/0/0.gz'
		sys.exit(0)
	ifname=sys.argv[1]
	color,depth,ptcloud,keynames=loadBinFile(ifname)
	cv2.imshow('color',cv2.resize(plotimggrid(color),(0,0),fx=0.5,fy=0.5))
	cv2.imshow('depth',cv2.resize(plotimggrid(np.uint8(depth>>8)[:,:,:,None]+[0,0,0]),(0,0),fx=0.5,fy=0.5))
	print 'press q to navigate next, Q to quit'
	for i in xrange(len(ptcloud)):
		print i,keynames[i]
		show3d.showpoints(ptcloud[i])
