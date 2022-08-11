import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from PointConv import feature_encoding_layer, feature_decoding_layer, pointnet_fp_module

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class,sigma, bn_decay=None, weight_decay = None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:,:,0:3]
    l0_points = point_cloud
    
    # Feature encoding layers
    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=2048,size=3,layers=1,stride=2, mlp=32, radius=0.04, sigma=0.5*sigma, relative_xyz=True,  is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    l1_xyz, l1b_points = feature_encoding_layer(l1_xyz, l1_points, npoint=2048,size=3,layers=1,stride=1,mlp=32, radius=0.04, sigma=0.5*sigma, relative_xyz=False, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1b')
    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1b_points, npoint=1024,size=3,layers=1,stride=2, mlp=64, radius=0.08, sigma=1*sigma, relative_xyz=False, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    l2_xyz, l2b_points = feature_encoding_layer(l2_xyz, l2_points, npoint=1024,size=3,layers=1,stride=1, mlp=64, radius=0.08, sigma=1*sigma, relative_xyz=False, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2b')
    l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2b_points, npoint=512, size=3,layers=1,stride=2,mlp=128, radius=0.16, sigma=2*sigma, relative_xyz=False, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    l3_xyz, l3b_points = feature_encoding_layer(l3_xyz, l3_points, npoint=512, size=3,layers=1,stride=1,mlp=128, radius=0.16, sigma=2*sigma, relative_xyz=False, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3b')
    l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3b_points, npoint=256,size=3,layers=1,stride=2, mlp=256, radius=0.32, sigma=4*sigma, relative_xyz=False, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')
    l4_xyz, l4b_points = feature_encoding_layer(l4_xyz, l4_points, npoint=256,size=3,layers=1,stride=1,mlp=256, radius=0.32, sigma=4*sigma, relative_xyz=False, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4b')
    l5_xyz, l5_points = feature_encoding_layer(l4_xyz, l4b_points, npoint=128,size=3,layers=1,stride=2, mlp=512, radius=0.64, sigma=8*sigma, relative_xyz=False, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer5')
    l5_xyz, l5b_points = feature_encoding_layer(l5_xyz, l5_points, npoint=128,size=3,layers=1,stride=1, mlp=512, radius=0.64, sigma=8*sigma, relative_xyz=False, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer5b')
   
    # Feature decoding layers
    l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4b_points, l5b_points, [512, 512], is_training, bn_decay, weight_decay, radius=0.64, last_layer=False, scope='fa_layer1')
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3b_points, l4_points, [256, 256], is_training, bn_decay, weight_decay, radius=0.32, last_layer=False, scope='fa_layer2')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2b_points, l3_points, [256, 256], is_training, bn_decay, weight_decay, radius=0.16, last_layer=False, scope='fa_layer3')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1b_points, l2_points, [256, 128], is_training, bn_decay, weight_decay, radius=0.08, last_layer=False, scope='fa_layer4')
    net = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128], is_training, bn_decay, weight_decay, radius=0.04, last_layer=True, scope='fa_layer5')

    # FC layers
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, weight_decay=weight_decay, scope='fc2')##

    return net, end_points


def get_loss(pred, label, smpw):
    """ pred: BxNxC,
        label: BxN,
	smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    weight_reg = tf.add_n(tf.get_collection('losses'))
    classify_loss_mean = tf.reduce_mean(classify_loss, name='classify_loss_mean')
    total_loss = classify_loss_mean + weight_reg
    tf.summary.scalar('classify loss', classify_loss)
    tf.summary.scalar('total loss', total_loss)
    return total_loss

if __name__=='__main__':
    import pdb
    pdb.set_trace()

    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True), 10,0.05)
        print(net)
