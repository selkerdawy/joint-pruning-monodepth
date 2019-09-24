from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
import pdb
import cv2
import math
import glob
import os

from model_inference import *
from monodepth_dataloader import *


parser = argparse.ArgumentParser(description='Sample code for Joint end-to-end pruning.')

parser.add_argument('--dir',             type=str,   help='root directory', required=False)
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)

args = parser.parse_args()
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp



def test_video(params):
    """Test function."""
    try:
        left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
        nw_nfilter = np.load('eigen_pruned.npy')
        model = MonodepthModel(params, "test", left, None,archFromNP=nw_nfilter)
        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # RESTORE
        restore_path = args.checkpoint_path.split(".")[0]
        train_saver.restore(sess, restore_path)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outvid = cv2.VideoWriter('output.avi',fourcc, 5.0, (1242,375*2))

        filenames = sorted(glob.glob(args.dir+'*.jpg'))
        tot_time = 0.0
        #warm up for lazy loading in cuda
        for i in range(5):
            input_image = scipy.misc.imread(filenames[0], mode="RGB")
            orig_input = input_image
            original_height, original_width, num_channels = input_image.shape
            input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
            input_image = input_image.astype(np.float32) / 255
            input_images = np.stack((input_image, np.fliplr(input_image)), 0)
            disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})

        for cur,filename in enumerate(filenames):
            input_image = scipy.misc.imread(filename, mode="RGB")
            orig_input = input_image
            original_height, original_width, num_channels = input_image.shape
            input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
            input_image = input_image.astype(np.float32) / 255
            input_images = np.stack((input_image, np.fliplr(input_image)), 0)

            start = time.time()
            disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
            end = time.time()
            tot_time += 0 if cur == 0 else (end-start)
            fps = "FPS: - "
            if cur > 0:
                print("FPS: %f"%(cur/tot_time))
                fps = "FPS: %d"%round(cur/tot_time)

            disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
            disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
            disp_to_img = cv2.applyColorMap(np.uint8(disp_to_img), cv2.COLORMAP_JET)

            outframe = np.vstack([cv2.cvtColor(orig_input,cv2.COLOR_RGB2BGR),disp_to_img])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(outframe,fps, (orig_input.shape[1]-200,orig_input.shape[0]+50), font, 1,(0,255,0),2,cv2.LINE_AA)
            outvid.write(outframe)

            print('Done %s'%filename)
    except:
        outvid.release()
        print('done!')

def main(_):

    params = monodepth_parameters(
        encoder='vgg',
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        distill_loss_weight=0,
        full_summary=False)

    test_video(params)

if __name__ == '__main__':
    tf.app.run()
