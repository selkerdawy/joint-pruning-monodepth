# Copyright (c) Sara Elkerdawy 2019


from __future__ import absolute_import, division, print_function
from collections import namedtuple

import pdb
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops

#from bilinear_sampler import *

monodepth_parameters = namedtuple('parameters',
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'distill_loss_weight, '
                        'lr_loss_weight, '
                        'full_summary')

class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, params, mode, left, right, reuse_variables=None, model_index=0,teacher_model=None,distill=None,isPrune=False,archFromNP=None):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.model_collection = ['model_' + str(model_index)]
        self.archFromNP = archFromNP
        self.cnt = 0
        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

    #helper function from monodepth framework
    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    #helper function from monodepth framework
    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    #helper function from monodepth framework
    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu,padding='VALID',with_prune=None,slice_inp=False):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        if stride > 0:
            p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
            out= slim.conv2d(p_x, num_out_layers, kernel_size, stride, padding, activation_fn=activation_fn)
        else:
            p_x = x
            out= slim.conv2d(p_x, num_out_layers, kernel_size, padding=padding, activation_fn=activation_fn)

        returnLst = [out]
        out2 = None
        if slice_inp:
            out,out2 = tf.split(out,2,0)#split batches dim0 into two splits
            returnLst = [out,out2]

        return returnLst if len(returnLst)>1 else returnLst[0]

    def conv_block(self, x, num_out_layers, kernel_size,with_prune=None,extra_inp=None):
        slice_inp = False
        if not extra_inp is None:
            x = tf.concat([x , extra_inp],0)
            slice_inp = True
        if self.archFromNP is None:
            conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
            conv2 = self.conv(conv1, num_out_layers, kernel_size, 2,with_prune=with_prune,slice_inp = slice_inp)
        else:
            nf = self.archFromNP[self.cnt]
            self.cnt +=1
            conv1 = self.conv(x, nf , kernel_size, 1)
            nf = self.archFromNP[self.cnt]
            self.cnt +=1
            conv2 = self.conv(conv1, nf, kernel_size, 2,with_prune=with_prune,slice_inp = slice_inp)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    #helper function from monodepth framework
    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    #helper function from monodepth framework
    def build_vgg_prune(self):
        #set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input,  32, 7) # H/2
            conv2 = self.conv_block(conv1,             64, 5) # H/4
            conv3 = self.conv_block(conv2,            128, 3) # H/8
            conv4 = self.conv_block(conv3,            256, 3) # H/16
            conv5 = self.conv_block(conv4,            512, 3) # H/32
            conv6 = self.conv_block(conv5,            512, 3) # H/64
            conv7 = self.conv_block(conv6,            512, 3) # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        with tf.variable_scope('decoder'):
            nf = self.archFromNP[self.cnt]
            self.cnt+=2
            upconv7 = upconv(conv7,  nf, 3, 2) #H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7  = conv(concat7,  512, 3, 1)

            nf = self.archFromNP[self.cnt]
            self.cnt+=2
            upconv6 = upconv(iconv7, nf, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,  512, 3, 1)

            nf = self.archFromNP[self.cnt]
            self.cnt+=2
            upconv5 = upconv(iconv6, nf, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,  256, 3, 1)

            nf = self.archFromNP[self.cnt]
            self.cnt+=3
            upconv4 = upconv(iconv5, nf, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,  128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4  = self.upsample_nn(self.disp4, 2)

            nf = self.archFromNP[self.cnt]
            self.cnt+=3
            upconv3 = upconv(iconv4,  nf, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,   64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3  = self.upsample_nn(self.disp3, 2)

            nf = self.archFromNP[self.cnt]
            self.cnt+=3
            upconv2 = upconv(iconv3,  nf, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,   32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2  = self.upsample_nn(self.disp2, 2)

            nf = self.archFromNP[self.cnt]
            self.cnt+=3
            upconv1 = upconv(iconv2,  nf, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],activation_fn=tf.nn.elu):#,weights_regularizer=slim.l1_regularizer(0.001)):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.left_pyramid  = self.scale_pyramid(self.left,  4)
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)

                if self.params.do_stereo:
                    self.model_input = tf.concat([self.left, self.right], 3)
                else:
                    self.model_input = self.left

                #build model
                if self.params.encoder == 'vgg':
                    self.build_vgg_prune()

    def epoch_op(self):
        self.epoch = self.epoch + 1
        return self.epoch

    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]

        if self.mode == 'test':
            return

