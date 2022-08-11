import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_findl_so.so'))
def findl(xyz):
    '''
    Input:
	xyz:(b,n,3)points after fartherestpointsampling
    Output:
	l:(b)mean fartherestdistance
    '''
    return grouping_module.findl(xyz)
ops.NoGradient('Findl')

if __name__=='__main__':
    import numpy as np
    import time
    np.random.seed(100)
    #pts = np.random.random((8,1024,3)).astype('float32')
    pts=[[[1.0,1.0,2.0],[3.0,2.0,2.0],[2.0,1.0,4.0]],[[2.0,1.0,5.0],[4.0,9.0,5.0],[1.0,2.0,1.0]]]
    with tf.device('/cpu:0'):
	points=tf.constant(pts)
	l=findl(points)
    with tf.Session('') as sess:
	now = time.time() 
	#for _ in range(100):
	ret = sess.run(l)
	print(time.time() - now)
	print(ret.shape, ret.dtype)
	print(l.eval())
