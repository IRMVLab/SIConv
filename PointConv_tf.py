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
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/ops'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/L'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/feature_density'))
from tf_interpolate import three_nn, three_interpolate
from tf_grouping_self import query_and_interpolation
from tf_findl import findl
from tf_feature_density import find_feature_density
from tf_grouping import query_ball_point
import pointconv_util
import tf_util

delta_xyz1 = [[0.0, 0.0, -1.0128], [0.5, 0.2887, -0.5064], [-0.5, 0.2887, -0.5064], [0.0, -0.5774, -0.5064], [0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0], [0.5, 0.8660, 0.0], [-0.5, 0.8660, 0.0], [-1.0, 0.0, 0.0], [-0.5, -0.8660, 0.0], [0.5,-0.8660, 0.0],
              [0.5, 0.2887, 0.5064], [-0.5, 0.2887, 0.5064], [0.0, -0.5774, 0.5064], [0.0, 0.0, 1.0128]]
delta_xyz2 = [[]]
delta_xyz3 = [[]]
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
        channel = feature.get_shape()[2]
        if num_points == npoint:
            new_xyz = xyz
        else:
            new_xyz, _ = pointconv_util.sampling(npoint, xyz)#(b, n, 3)

        if layers == 1:
            lamda = 15
            delta_xyz = tf.tile(delta_xyz1, [npoint, 1])
        elif layers == 2:
            lamda = 65
            delta_xyz = tf.tile(delta_xyz2, [npoint, 1])
        else:
            lamda = 175
            delta_xyz = tf.tile(delta_xyz3, [npoint, 1])
        K=8
        #delta_xyz = tf.zeros([batch_size, npoint*lamda, 3])
        cell_xyz = tf.keras.backend.repeat_elements(new_xyz, rep=lamda, axis=1) - delta_xyz*radius#(b, n * 15, 3)
        n = cell_xyz.get_shape()[1]
        point_idx, pts_cnt = query_ball_point(radius, 4*K, xyz, cell_xyz)#point_idx(b, n*15, 4*K), pts_cnt(b, n*15)
        batch_idx_nk = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, n, 4*K, 1))
        idx = tf.concat([batch_idx_nk, tf.expand_dims(point_idx, axis=3)], axis=3)
        #grouped_nk_xyz = tf.gather_nd(xyz, idx)#(b, n*15, 4*K, 3)
        #grouped_xyz = tf.reshape(grouped_nk_xyz, [batch_size*n, -1, 3])#(b*n*15, 4*K, 3)
        grouped_feature1 = tf.gather_nd(feature, idx)
        grouped_feature = grouped_feature1[:, :, 0:K, :]

        # _, idx_nk_k = pointconv_util.sampling(K, grouped_xyz)#(b*n*15, K)
        # idx_k = tf.reshape(idx_nk_k, [batch_size, -1, K])#(b, n*15, K)
        # batch_idx_k = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, n, K, 1))
        # feat_idx = tf.concat([batch_idx_k, tf.expand_dims(idx_k, axis=3)], axis=3)
        # grouped_feature = tf.gather_nd(feature, feat_idx)  # (b, n*15, K, c)


        new_points1 = tf_util.conv2d(grouped_feature, mlp[0], kernel_size=[1, grouped_feature.get_shape()[2].value],
                                    padding='VALID', data_format='NHWC', stride=[1, 1],
                                    bn=bn, is_training=is_training,
                                    scope=scope+'1', bn_decay=bn_decay, weight_decay=weight_decay)
        new_points2 = tf.squeeze(new_points1, [2])
        new_points3 = tf.reshape(new_points2, [batch_size, npoint, lamda, -1])
        new_points4 = tf_util.conv2d(new_points3, mlp[-1], kernel_size=[1, new_points3.get_shape()[2].value],
                                     padding='VALID', data_format='NHWC', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope=scope+'2', bn_decay=bn_decay, weight_decay=weight_decay)

        new_points = tf.squeeze(new_points4, [2])

        # grouped_feature1 = tf.reshape(grouped_feature, [batch_size, npoint, lamda, K, channel])
        # new_points1 = tf_util.conv3d(grouped_feature1, mlp, kernel_size=[1, lamda, K],
        #                              padding='VALID', data_format='NDHWC', stride=[1, 1, 1],
        #                              bn=bn, is_training=is_training,
        #                              scope=scope, bn_decay=bn_decay, weight_decay=weight_decay)
        #
        # new_points = tf.squeeze(new_points1, [2, 3])

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
        npoint1 = xyz1.get_shape()[1]
        batch_size=xyz1.get_shape()[0]
        channel = points2.get_shape()[2]
        K=8
        if layers == 1:
            lamda = 15
            delta_xyz = tf.tile(delta_xyz1, [npoint1, 1])
        elif layers == 2:
            lamda = 65
            delta_xyz = tf.tile(delta_xyz2, [npoint1, 1])
        else:
            lamda = 175
            delta_xyz = tf.tile(delta_xyz3, [npoint1, 1])
        cell_xyz = tf.keras.backend.repeat_elements(xyz1, rep=lamda, axis=1) - delta_xyz * radius # (b, n * 15, 3)
        n = cell_xyz.get_shape()[1]
        point_idx, pts_cnt = query_ball_point(radius, 4 * K, xyz2, cell_xyz)  # point_idx(b, n*15, 4*K), pts_cnt(b, n*15)
        batch_idx_nk = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, n, 4 * K, 1))
        idx = tf.concat([batch_idx_nk, tf.expand_dims(point_idx, axis=3)], axis=3)
        #grouped_nk_xyz = tf.gather_nd(xyz2, idx)  # (b, n*15, 4*K, 3)
        grouped_feature1 = tf.gather_nd(points2, idx)
        grouped_feature = grouped_feature1[:, :, 0:K, :]#(b, n*15, K, c)

        # grouped_xyz = tf.reshape(grouped_nk_xyz, [batch_size * n, -1, 3])  # (b*n*15, 4*K, 3)
        # _, idx_nk_k = pointconv_util.sampling(K, grouped_xyz)  # (b*n*15, K)
        # idx_k = tf.reshape(idx_nk_k, [batch_size, -1, K])  # (b, n*15, K)
        # batch_idx_k = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, n, K, 1))
        # feat_idx = tf.concat([batch_idx_k, tf.expand_dims(idx_k, axis=3)], axis=3)
        # grouped_feature = tf.gather_nd(points2, feat_idx)  # (b, n*15, K, c)

        new_points1 = tf_util.conv2d(grouped_feature, mlp[0], kernel_size=[1, grouped_feature.get_shape()[2].value],
                                    padding='VALID', data_format='NHWC', stride=[1, 1],
                                    bn=bn, is_training=is_training,
                                    scope=scope+'1', bn_decay=bn_decay, weight_decay=weight_decay)
        new_points2 = tf.squeeze(new_points1, [2])
        new_points3 = tf.reshape(new_points2, [batch_size, npoint1, lamda, -1])
        points1_concat = tf.tile(tf.expand_dims(points1, axis=2), [1, 1, lamda, 1])
        new_points4 = tf_util.conv2d(tf.concat([new_points3, points1_concat], axis=-1), mlp[-1], kernel_size=[1, new_points3.get_shape()[2].value],
                                     padding='VALID', data_format='NHWC', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope=scope+'2', bn_decay=bn_decay, weight_decay=weight_decay)

        new_points = tf.squeeze(new_points4, [2])

        # grouped_feature1 = tf.reshape(grouped_feature, [batch_size, npoint1, lamda, K, channel])
        # new_points1 = tf_util.conv3d(grouped_feature1, mlp, kernel_size=[1, lamda, K],
        #                              padding='VALID', data_format='NDHWC', stride=[1, 1, 1],
        #                              bn=bn, is_training=is_training,
        #                              scope=scope, bn_decay=bn_decay, weight_decay=weight_decay)
        # new_points = tf.squeeze(new_points1, [2, 3])

        return new_points

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





