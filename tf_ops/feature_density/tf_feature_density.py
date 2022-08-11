import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_feature_density_so.so'))

def find_feature_density(xyz,feature,radius):
	'''
    Input:
        xyz: (batch_size, ndataset, 3) float32 array, input points
		feature:(batch_size,ndataset,1)float32 array
		radius:(b)radius
    Output:
        feature_density:(batch_size, ndataset, 1)
    '''
	return grouping_module.find_feature_density(xyz,feature,radius)
ops.NoGradient('FindFeatureDensity')
#@tf.RegisterGradient('FindFeatureDensity')
"""
def find_feature_density_grad(op,grad_out):
    xyz = op.inputs[0]
    feature = op.inputs[1]
    radius = op.inputs[2]
    return [None, grouping_module.find_feature_density_grad(xyz, feature, radius, grad_out), None]
"""


if __name__=='__main__':
    import numpy as np
    import time
    np.random.seed(100)
    pts = np.random.random((32,512,64)).astype('float32')
    #tmp1 = np.random.random((32,512,3)).astype('float32')
    #tmp2 = np.random.random((32,512,1)).astype('float32')
    tmp1=[[[1.0,1.0,2.0],[2.0,2.0,2.0],[1.0,1.0,2.0]]]
    tmp2=[[[1.0],[2.0],[3.0]]]
    with tf.device('/gpu:1'):
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        feature = tf.constant(tmp2)
        radius = [3.0] 
        feature_density=find_feature_density(xyz1,feature,radius)
        
        #grouped_points_grad = tf.ones_like(grouped_points)
        #points_grad = tf.gradients(grouped_points, points, grouped_points_grad)
    with tf.Session('') as sess:
        now = time.time() 
        for _ in range(100):
            ret = sess.run(feature_density)
        print(time.time() - now)
        print(ret.shape, ret.dtype)
        print(ret)
