"""
PointConv operation
Author: Wenxuan Wu
Date: July 2018
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/ops_square'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/L'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/feature_density'))
from tf_interpolate import three_nn, three_interpolate
from tf_grouping_self import query_and_interpolation
from tf_findl import findl
from tf_feature_density import find_feature_density
import tf_grouping
import pointconv_util
import tf_util

def weight_net_hidden(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn=tf.nn.relu):

    with tf.variable_scope(scope) as sc:
        net = xyz
        for i, num_hidden_units in enumerate(hidden_units):
            net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                padding = 'VALID', stride=[1, 1],
                                bn = True, is_training = is_training, activation_fn=activation_fn,
                                scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

            #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='wconv_dp%d'%(i))
    return net

def weight_net(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn=tf.nn.relu):

    with tf.variable_scope(scope) as sc:
        net = xyz
        for i, num_hidden_units in enumerate(hidden_units):
            if i != len(hidden_units) -1:
                net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                    padding = 'VALID', stride=[1, 1],
                                    bn = True, is_training = is_training, activation_fn=activation_fn,
                                    scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
            else:
                net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                    padding = 'VALID', stride=[1, 1],
                                    bn = False, is_training = is_training, activation_fn=None,
                                    scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
            #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='wconv_dp%d'%(i))
    return net

def nonlinear_transform(data_in, mlp, scope, is_training, bn_decay=None, weight_decay = None, activation_fn = tf.nn.relu):

    with tf.variable_scope(scope) as sc:

        net = data_in
        l = len(mlp)
        if l > 1:
            for i, out_ch in enumerate(mlp[0:(l-1)]):
                net = tf_util.conv1d(net, out_ch, 1,
                                    padding = 'VALID', stride=1,
                                    bn = True, is_training = is_training, activation_fn=tf.nn.relu,
                                    scope = 'nonlinear%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

                #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp_nonlinear%d'%(i))
        net = tf_util.conv1d(net, mlp[-1], 1,
                            padding = 'VALID', stride=1,
                            bn = False, is_training = is_training,
                            scope = 'nonlinear%d'%(l-1), bn_decay=bn_decay,
                            activation_fn=tf.nn.sigmoid, weight_decay = weight_decay)

    return net

def feature_encoding_layer1(xyz, feature, npoint, mlp, is_training, bn_decay, weight_decay, scope, bn=True, use_xyz=True):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            sigma: float32 -- KDE bandwidth
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        num_points = xyz.get_shape()[1]
        if num_points == npoint:
            new_xyz = xyz
        else:
            new_xyz = pointconv_util.sampling(npoint, xyz)

        l=2*findl(new_xyz)
        weight_space=query_and_interpolation(xyz, feature, new_xyz, l,5)#(b,npoints,125,c)

        batch_size = weight_space.get_shape()[0]
        npoint = weight_space.get_shape()[1]
        c=weight_space.get_shape()[3]

        grouped_feature=tf.reshape(weight_space,(batch_size,npoint*5,5,5,c))
        for i, num_out_channel in enumerate(mlp):
            if i != len(mlp) - 1:
                grouped_feature = tf_util.conv3d(grouped_feature, num_out_channel, kernel_size=[1,1,1],
                                            padding='VALID', stride=[1,1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
        new_points=tf_util.conv3d(grouped_feature, mlp[-1], kernel_size=[5,5,5],
                                padding='VALID', stride=[5,1,1],
                                bn=bn, is_training=is_training,
                                scope='after_conv', bn_decay=bn_decay, weight_decay = weight_decay)

        new_points= tf.squeeze(new_points, [2,3]) #(b,n,c)
       
	return new_xyz, new_points


def feature_encoding_layer(xyz, feature, npoint,size,layers,stride, mlp, radius, sigma, is_training, bn_decay, weight_decay, scope, bn=True, use_xyz=True):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            sigma: float32 -- KDE bandwidth
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        num_points = xyz.get_shape()[1]
        batch_size = xyz.get_shape()[0]
        if num_points == npoint:
            new_xyz = xyz
        else:
            new_xyz = pointconv_util.sampling(npoint, xyz)

        if npoint>=32:
            K=32
        else:
            K=8
        """
        #_, _, idx = pointconv_util.grouping(feature, K, xyz, xyz)
        #l = size * findl(new_xyz) / stride  # (b)
        l_cell = tf.tile(tf.reshape(radius,(1,)),[batch_size])
        #print(l)
        l = l_cell# (b)
        dist_density = pointconv_util.kernel_density_estimation_ball(xyz, xyz, l_cell, sigma, is_norm=True)
        feat_density = pointconv_util.kernel_density_estimation_ball(xyz, feature, l_cell, sigma, is_norm=True)
        # inverse_density = tf.div(1.0, density)
        # grouped_dist_density = tf.gather_nd(dist_density, idx)  # (batch_size, npoint, nsample, 1)
        # grouped_feat_density = tf.gather_nd(feat_density, idx)  # (batch_size, npoint, nsample, 1)
        # grouped_density = tf_grouping.group_point(inverse_density, idx)
        grouped_density = tf.concat([dist_density, feat_density], axis=-1)
        # max_density = tf.reduce_max(grouped_density, axis=2, keepdims=True)
        # density_scale = tf.div(grouped_density, max_density)
        feature_density = nonlinear_transform(grouped_density, [16, 1], scope='density_net', is_training=is_training,
                                              bn_decay=bn_decay, weight_decay=weight_decay)
        # feature_density = tf.squeeze(feature_density, [2])
        # feature_density = tf.reduce_sum(feature_density,axis=2)
        

        feature_density = find_feature_density(xyz, feature_density, l_cell)
        """
        l_cell = tf.tile(tf.reshape(radius, (1,)), [batch_size])
        l = l_cell
        feature_density = tf.ones([batch_size, num_points, 1])

        weight_space=query_and_interpolation(xyz, feature, new_xyz,feature_density, l,size)#(b,npoints,125,c)


        new_points=tf_util.conv2d(weight_space, mlp, kernel_size=[1,weight_space.get_shape()[2].value],
                    padding='VALID',data_format='NHWC', stride=[1,1],
                    bn=bn, is_training=is_training,
                    scope=scope, bn_decay=bn_decay, weight_decay = weight_decay)

        new_points=tf.squeeze(new_points, [2]) #(b,n,c)

        return new_xyz, new_points

def feature_decoding_layer(xyz1, points1,xyz2, points2, size,layers, stride, mlp, radius, sigma, is_training, bn_decay, weight_decay, scope, bn=True, use_xyz = True):
    ''' Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            sigma: float32 -- KDE bandwidth
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        npoint=xyz2.get_shape()[1]
        batch_size=xyz2.get_shape()[0]
        if npoint>=16:
            K=16
        else:
            K=8
        """
        #_, _, idx = pointconv_util.grouping(points2, K, xyz2, xyz2, use_xyz=use_xyz)
        l_cell = tf.tile(tf.reshape(radius, (1,)), [batch_size])
        l = l_cell# (b)
        dist_density = pointconv_util.kernel_density_estimation_ball(xyz2, xyz2, l_cell, sigma, is_norm=True)
        feat_density = pointconv_util.kernel_density_estimation_ball(xyz2, points2, l_cell, sigma, is_norm=True)
        # inverse_density = tf.div(1.0, density)
        # grouped_dist_density = tf.gather_nd(dist_density, idx)  # (batch_size, npoint, nsample, 1)
        # grouped_feat_density = tf.gather_nd(feat_density, idx)  # (batch_size, npoint, nsample, 1)
        # grouped_density = tf_grouping.group_point(inverse_density, idx)
        grouped_density = tf.concat([dist_density, feat_density], axis=-1)
        # max_density = tf.reduce_max(grouped_density, axis=2, keepdims=True)
        # density_scale = tf.div(grouped_density, max_density)
        feature_density = nonlinear_transform(grouped_density, [16, 1], scope='density_net', is_training=is_training,
                                              bn_decay=bn_decay, weight_decay=weight_decay)
        # feature_density = tf.squeeze(feature_density, [2])
        # feature_density = tf.reduce_sum(feature_density,axis=2)

        feature_density = find_feature_density(xyz2, feature_density, l_cell)
        """

        l_cell = tf.tile(tf.reshape(radius, (1,)), [batch_size])
        l = l_cell
        feature_density = tf.ones([batch_size, npoint, 1])

        weight_space=query_and_interpolation(xyz2,points2,xyz1,feature_density,l,size)#(b,ndataset1,27,c)

        new_points = tf_util.conv2d(weight_space, mlp, [1,weight_space.get_shape()[2].value],
                                        padding='VALID',data_format='NHWC', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope=scope, bn_decay=bn_decay, weight_decay = weight_decay) 

        new_points = tf.squeeze(new_points, [2]) # B,ndataset1,mlp[-1]
#
        return new_points


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2, keepdims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1


def feature_decoding_layer_depthwise(xyz1, xyz2, points1, points2, radius, sigma, K, mlp, is_training, bn_decay, weight_decay, scope, bn=True, use_xyz = True):
    ''' Input:                                      
            depthwise version of pointconv                                                                
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            sigma: float32 -- KDE bandwidth
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keepdims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        #setup for deConv
        grouped_xyz, grouped_feature, idx = pointconv_util.grouping(interpolated_points, K, xyz1, xyz1, use_xyz=use_xyz)

        density = pointconv_util.kernel_density_estimation_ball(xyz1, radius, sigma)
        inverse_density = tf.div(1.0, density)
        grouped_density = tf.gather_nd(inverse_density, idx) # (batch_size, npoint, nsample, 1)
        #grouped_density = tf_grouping.group_point(inverse_density, idx)
        inverse_max_density = tf.reduce_max(grouped_density, axis = 2, keepdims = True)
        density_scale = tf.div(grouped_density, inverse_max_density)

        #density_scale = tf_grouping.group_point(density, idx)

        weight = weight_net(grouped_xyz, [32, grouped_feature.get_shape()[3].value], scope = 'decode_weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)

        density_scale = nonlinear_transform(density_scale, [16, 1], scope = 'decode_density_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)

        new_points = tf.multiply(grouped_feature, density_scale)

        new_points = tf.multiply(grouped_feature, weight)

        new_points = tf_util.reduce_sum2d_conv(new_points, axis = 2, scope = 'fp_sumpool', bn=True,
                                        bn_decay = bn_decay, is_training = is_training, keepdims = False)    

        if points1 is not None:
            new_points1 = tf.concat(axis=-1, values=[new_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = new_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1

def placeholder_inputs(batch_size, num_point, channel):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    feature_pts_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, channel))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, feature_pts_pl, labels_pl

if __name__=='__main__':
    import numpy as np
    pts = np.random.random((32, 2048, 3)).astype('float32')
    fpts = pts
    sigma = 0.1
    N = 512
    K = 64
    D = 1
    C_list = [64, 128]
    mlp_w = [64]
    mlp_d = [64]
    is_training = tf.placeholder(tf.bool, shape=())

    import pdb
    pdb.set_trace()

    with tf.device('/gpu:1'):
        #points = tf.constant(pts)
        #features = tf.constant(fpts)
        points_pl, features_pl, labels_pl = placeholder_inputs(32, 8192, 3)
        sub_pts, features = feature_encoding_layer(points_pl, features_pl, N, 3,2, 10, is_training, bn_decay = 0.1, weight_decay = 0.1, scope = "FE")
        feature_decode = feature_decoding_layer(points_pl, sub_pts, features, 3,2, 23, is_training, bn_decay=0.1, weight_decay = 0.1, scope= "FD")





