import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from PointConv_without_rgb import pointnet_fp_module, feature_encoding_layer, feature_decoding_layer

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
#    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=1024, radius = 0.1, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
#    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
#    l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius = 0.4, sigma = 4 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
#    l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3_points, npoint=36, radius = 0.8, sigma = 8 * sigma, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')

    # Feature decoding layers
#    l3_points = feature_decoding_layer(l3_xyz, l4_xyz, l3_points, l4_points, 0.8, 8 * sigma, 16, [512,512], is_training, bn_decay, weight_decay, scope='fa_layer1')
#    l2_points = feature_decoding_layer(l2_xyz, l3_xyz, l2_points, l3_points, 0.4, 4 * sigma, 16, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2')
#    l1_points = feature_decoding_layer(l1_xyz, l2_xyz, l1_points, l2_points, 0.2, 2 * sigma, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
#    l0_points = feature_decoding_layer(l0_xyz, l1_xyz, l0_points, l1_points, 0.1, sigma, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')
    # Feature encoding layers
    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=2048,size=3,layers=1,stride=2, mlp=32, radius=0.05, sigma=0.5*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    l1_xyz, l1b_points = feature_encoding_layer(l1_xyz, l1_points, npoint=2048,size=3,layers=1,stride=1,mlp=32, radius=0.05, sigma=0.5*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1b')
    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1b_points, npoint=1024,size=3,layers=1,stride=2, mlp=64, radius=0.1, sigma=sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    l2_xyz, l2b_points = feature_encoding_layer(l2_xyz, l2_points, npoint=1024,size=3,layers=1,stride=1, mlp=64, radius=0.1, sigma=sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2b')

    #l2i_xyz, l2bi_points = feature_encoding_layer(l2_xyz, l2b_points, npoint=256,size=5,stride=1, mlp=64, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2bi')

    l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2b_points, npoint=512, size=3,layers=1,stride=2,mlp=128, radius=0.2, sigma=2*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    l3_xyz, l3b_points = feature_encoding_layer(l3_xyz, l3_points, npoint=512, size=3,layers=1,stride=1,mlp=128, radius=0.2, sigma=2*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3b')
    l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3b_points, npoint=256,size=3,layers=1,stride=2, mlp=256, radius=0.4, sigma=4*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')
    l4_xyz, l4b_points = feature_encoding_layer(l4_xyz, l4_points, npoint=256,size=3,layers=1,stride=1,mlp=256, radius=0.4, sigma=4*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4b')
    l5_xyz, l5_points = feature_encoding_layer(l4_xyz, l4b_points, npoint=128,size=3,layers=1,stride=2, mlp=512, radius=0.8, sigma=8*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer5')
    l5_xyz, l5b_points = feature_encoding_layer(l5_xyz, l5_points, npoint=128,size=3,layers=1,stride=1, mlp=512, radius=0.8, sigma=8*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer5b')
    #m = l5b_points.get_shape()[1]
    #l8_points = tf.nn.pool(input=l5b_points, window_shape=[m], pooling_type='MAX', padding='VALID')
    #l8_points = tf.tile(l8_points,[1,8192,1])
    #l6_xyz,l6_points = feature_encoding_layer(l5_xyz, l5b_points, npoint=4,size=3,stride=2, mlp=1024, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer6')
    #l6_xyz, l6b_points = feature_encoding_layer(l6_xyz, l6_points, npoint=4,size=3,stride=1, mlp=1024, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer6b')
    # Feature decoding layers
    
    #l6_points_up = feature_decoding_layer(l5_xyz, l6_xyz, l6b_points,3,2,1024, is_training, bn_decay, weight_decay, scope='up_layer6')
    #l5_xyz, l5b_points_i = feature_encoding_layer(l5_xyz, tf.concat([l6_points_up,l5b_points],axis=2), npoint=16,size=3,stride=1, mlp=1024, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='i_layer6')
    """
    l5_points_up = feature_decoding_layer(l4_xyz, l4b_points,l5_xyz, l5b_points,3,1,2,512, 1.6, 16*sigma, is_training, bn_decay, weight_decay, scope='up_layer5')
    l4_xyz, l4b_points_i = feature_encoding_layer(l4_xyz, tf.concat([l5_points_up,l4b_points],axis=2), npoint=256,size=3,layers=1,stride=1, mlp=512, radius=1.6, sigma=16*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='i_layer5')

    l4_points_up = feature_decoding_layer(l3_xyz,l3b_points, l4_xyz, l4b_points_i,3,1,2,256, 0.8, 8*sigma,  is_training, bn_decay, weight_decay, scope='up_layer4')
    l3_xyz, l3b_points_i = feature_encoding_layer(l3_xyz, tf.concat([l4_points_up,l3b_points],axis=2), npoint=512,size=3,layers=1,stride=1, mlp=256, radius=0.8, sigma=8*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='i_layer4')

    l3_points_up = feature_decoding_layer(l2_xyz,l2b_points, l3_xyz, l3b_points_i,3,1,2,128, 0.4, 4*sigma, is_training, bn_decay, weight_decay, scope='up_layer3')
    l2_xyz, l2b_points_i = feature_encoding_layer(l2_xyz, tf.concat([l3_points_up,l2b_points],axis=2), npoint=1024,size=3,layers=1,stride=1, mlp=128, radius=0.4, sigma=4*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='i_layer3')

    l2_points_up = feature_decoding_layer(l1_xyz,l1b_points, l2_xyz, l2b_points_i,3,1,2,64, 0.2, 2*sigma, is_training, bn_decay, weight_decay, scope='up_layer2')
    l1_xyz, l1b_points_i = feature_encoding_layer(l1_xyz, tf.concat([l2_points_up,l1b_points],axis=2), npoint=2048,size=3,layers=1,stride=1, mlp=64, radius=0.2, sigma=2*sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='i_layer2')

    l1_points_up = feature_decoding_layer(l0_xyz, l0_points,l1_xyz, l1b_points_i,3,1,2,32, 0.1, sigma, is_training, bn_decay, weight_decay, scope='up_layer1')
    l0_xyz, net = feature_encoding_layer(l0_xyz, tf.concat([l1_points_up,l0_points],axis=2), npoint=8192,size=3,layers=1,stride=1, mlp=32, radius=0.1, sigma=sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='i_layer1')
    #l2_points = feature_decoding_layer(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2')
    #l1_points = feature_decoding_layer(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
    #l0_points = feature_decoding_layer(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')
    """
    l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4b_points, l5b_points, [512, 512], is_training, bn_decay, scope='fa_layer1')
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3b_points, l4_points, [256, 256], is_training, bn_decay, scope='fa_layer2')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2b_points, l3_points, [256, 256], is_training, bn_decay, scope='fa_layer3')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1b_points, l2_points, [256, 128], is_training, bn_decay, scope='fa_layer4')
    net = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128], is_training, bn_decay, scope='fa_layer5')

    # FC layers
    #l0_xyz, net = feature_encoding_layer(l0_xyz,l0_points_i, 8192,3,1,1, 32,  0.1, sigma, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='out')
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
