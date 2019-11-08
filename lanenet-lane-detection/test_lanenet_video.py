# @Time    : 25/10/2019
# @Author  : Khai Do

"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time
import tqdm
import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import os.path
from os import path

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from tools import test_lanenet

CFG = global_config.cfg



def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='The video path or the src video save dir', 
                        default = "data/tusimple_test_image/5.mp4")
    parser.add_argument('--weights_path', type=str, help='The model weights path' , default = 'model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt')
    parser.add_argument('--save_dir', type=str, help='Test result video save dir' , default = "data/output")
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet_batch(video_path, weights_path, use_gpu, save_dir=None):
    """

    :param video_path:
    :param weights_path:
    :param use_gpu:
    :param save_dir:
    :return:
    """

  
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)
    
    assert ops.exists(video_path), '{:s} not exist'.format(video_path)
    print('Protecting Video {}...'.format(video_path))
    
    
    t_start = time.time()
    # create video capture and get video info
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    output_dir = os.path.join(save_dir, 'output.mp4')
#        
    out = cv2.VideoWriter(output_dir , fourcc, fps, (1280,720))


    assert vidcap.isOpened(), 'Cannot capture source'
#
    frames = 0      

    with sess.as_default():
        
       
        while vidcap.isOpened():
            success, frame = vidcap.read()
            
            if success:
                t_start = time.time()
                image_ori2 = cv2.resize(frame, (1280,720))
                image_vis = cv2.resize(frame, (512, 256))
                image = cv2.resize(frame, (512, 256), interpolation=cv2.INTER_LINEAR)
                
                image = image / 127.5 - 1.0
        
                saver.restore(sess=sess, save_path=weights_path)
                t_start = time.time()
                binary_seg_image, instance_seg_image = sess.run(
                    [binary_seg_ret, instance_seg_ret],
                    feed_dict={input_tensor: [image]}
                )
                t_cost = time.time() - t_start
                log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))
        
                postprocess_result = postprocessor.postprocess(
                    binary_seg_result=binary_seg_image[0],
                    instance_seg_result=instance_seg_image[0],
                    source_image=image_vis
                )
                mask_image = postprocess_result['mask_image']
        
                for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
                    instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
                
                mask_image1 = cv2.resize(mask_image, (1280,720))
                if save_dir is not None:
                    mask_image2 = cv2.addWeighted(image_ori2, 1.0, mask_image1, 1.0, 0)
                    
                cv2.imshow("frame", mask_image2)
                out.write(mask_image2)
                
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                frames += 1
            else:
                break 
        vidcap.release()
        out.release()
        cv2.destroyAllWindows()
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    cur_dir = os.getcwd()
    test_lanenet_batch(video_path=os.path.join(cur_dir, args.video_path), weights_path=os.path.join(cur_dir, args.weights_path),
                           save_dir=os.path.join(cur_dir, args.save_dir), use_gpu=args.use_gpu)
