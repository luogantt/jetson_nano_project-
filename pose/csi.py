#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:17:29 2021

@author: ledi
"""

import cv2
 
 
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
 
 
# def show_camera():
#     cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
 
#     while cap.isOpened():
#         flag, img = cap.read()
#         cv2.imshow("CSI Camera", img)
#         kk = cv2.waitKey(1)
 
#         # do other things
 
#         if kk == ord('q'):  # 按下 q 键，退出
#             break
 
#     cap.release()
#     cv2.destroyAllWindows()
    
    


#from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints



ctx = mx.cpu()
detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)




detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
detector.hybridize()



estimator = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
estimator.hybridize()

#cap = cv2.VideoCapture(0)
time.sleep(1)  ### letting the camera autofocus



axes = None
num_frames = 100


cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
 
while cap.isOpened():
    flag, frame = cap.read()
    print(11111)
    # cv2.imshow("CSI Camera", img)
    # kk = cv2.waitKey(1)
 
    # # do other things
 
    # if kk == ord('q'):  # 按下 q 键，退出
    #     break

    # ret, frame = cap.read()
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    
    print(2222)

    x, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=350)
    
    print(333)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)
    print(444)

    pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                       output_shape=(128, 96), ctx=ctx)
    
    print(pose_input)
    if len(upscale_bbox) > 0:
        predicted_heatmap = estimator(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

        img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                box_thresh=0.5, keypoint_thresh=0.2)
        cv_plot_image(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 存储图片
        cv2.imwrite("camera.jpeg", frame)
        break

#python cam_demo.py --num-frames 100




