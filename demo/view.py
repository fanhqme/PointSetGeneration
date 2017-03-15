import show3d
import numpy as np
import sys
a=np.loadtxt(sys.argv[1])
show3d.showpoints(a)
