#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:29:25 2020

@author: lg
"""










import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

#from insight import insight_frame
#transform2=transforms.Compose([transforms.ToTensor()])
#tensor2=transform2(frame)

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
 
 

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
 

    



cap = cv2.VideoCapture(0)
import time
while(1):
    # 获得图片
    ret, frame = cap.read()
    # 展示图片
#    cv2.imshow("capture", frame)
#    tensor2=transform2(frame)
    #image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    image=frame

    cv2.imshow("OpenCV",image)



    # Add to frame list
#    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 存储图片
        cv2.imwrite("camera.jpeg", frame)
        break

cap.release()
cv2.destroyAllWindows()





"""

frames_tracked = []
for i, frame in enumerate(frames):
    print('\rTracking frame: {}'.format(i + 1), end='')
    
    # Detect faces
    boxes, _ = mtcnn.detect(frame)
    
    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    
    # Add to frame list
    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
print('\nDone')


# #### Display detections



d = display.display(frames_tracked[0], display_id=True)
i = 1
try:
    while True:
        d.update(frames_tracked[i % len(frames_tracked)])
        i += 1
except KeyboardInterrupt:
    pass


# #### Save tracked video



dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()


"""



