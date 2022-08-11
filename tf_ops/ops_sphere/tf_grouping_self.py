import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_so.so'))
def query_and_interpolation(xyz1,feature,xyz2,feature_density,lh,layers,relative_xyz=False):
    '''
    Input:
        lh:(b) float32, search space
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
	size:(1)size of kernel
    Output:
        space_weight:(batch_size,npoint,27,3)
    '''
    return grouping_module.query_and_interpolation_sphere(xyz1,feature,xyz2,feature_density,lh,layers,relative_xyz)
@tf.RegisterGradient('QueryAndInterpolationSphere')
def _query_and_interpolation_sphere_grad(op,grad_out):
    xyz1 = op.inputs[0]
    feature = op.inputs[1]
    xyz2 = op.inputs[2]
    feature_density = op.inputs[3]
    lh = op.inputs[4]
    layers = op.get_attr('layers')
    return [None, grouping_module.query_and_interpolation_sphere_grad(xyz1, feature, xyz2, feature_density, lh, grad_out,layers), None, None, None]


if  __name__=='__main__':
    import numpy as np
    import time
    np.random.seed(100)
    #tmp1 = np.random.random((32,512,3)).astype('float32')
    #tmp2 = np.random.random((32,128,3)).astype('float32')
    #tmp3 = np.random.random((32,512,4)).astype('float32')
    tmp1 = [[[1.0, 1.0, 7.0], [1.0, 10.1, 7.0]], [[2.0,2.0,1.0], [2.0, 20.1, 1.0]]]
    tmp2 = [[[1.0, 1.0, 7.0]], [[2.0, 2.0, 1.0]]]
    tmp3 = [[[3.0], [2.0]], [[3.0], [1.0]]]
    tmp4 = [[[1.0], [1.0]], [[1.0], [1.0]]]
    with tf.device('/gpu:0'):
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        feature=tf.constant(tmp3)
        feature_density = tf.constant(tmp4)
        lh_ = [2.0,0.6]
        lh = tf.constant(lh_)
        #size=5
        space_weight=query_and_interpolation(xyz1, feature, xyz2, feature_density,lh,1)
        with tf.Session('') as sess:
            now = time.time()
            for _ in range(100):
                ret = sess.run(space_weight)
            print(time.time() - now)
            print(ret.shape, ret.dtype)
            print(ret)
