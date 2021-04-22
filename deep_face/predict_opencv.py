#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:32:07 2020

@author: lg
"""

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


mtcnn1 = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,keep_all=False,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)


def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('./data/test_images')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
aligned = []
names = []
for x, y in loader:
    print(x,y)
    x_aligned, prob = mtcnn1(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])
        
aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names))



import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
#transform2=transforms.Compose([transforms.ToTensor()])
#tensor2=transform2(frame)



#mtcnn2 = MTCNN(
#    image_size=160, margin=0, min_face_size=20,
#    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,keep_all=True,
#    device=device
#)


def operate_frame(frame):
    
    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    temp, prob = mtcnn1(image, return_prob=True)
    #print(prob)
#    print(len(temp))
    if temp is not None and prob>0.8:
         
        c=min(temp.shape)
        print(c,'##################')

        if c>0:
            t1=torch.stack([temp]).to(device)
            t2=resnet(t1).detach().cpu()
            #print(len(t2))
            #time.sleep(0.2)
            #print(len(t2),'dim')
    #        if len(t2)>0:
    #            
    #            for k in t2:
            distenct=[[(e1 - e2).norm().item() for e2 in embeddings] for e1 in t2]
            
            dist1=np.array(distenct)
            d2=np.argmin(dist1, axis=1)
            
            print('face dist',dist1[0][d2[0]])
            if dist1[0][d2[0]]<1.8:
    #            print('you are',names[d2[0]])
                
                return names[d2[0]]



"""

cap = cv2.VideoCapture(0)
import time
while(1):
    # 获得图片
    ret, frame = cap.read()
    # 展示图片
    cv2.imshow("capture", frame)
#    tensor2=transform2(frame)
    
    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    temp, prob = mtcnn1(image, return_prob=True)
    #print(prob)
#    print(len(temp))
    if temp is not None and prob>0.8:
        t1=torch.stack([temp]).to(device)
        t2=resnet(t1).detach().cpu()
        print(len(t2))
        #time.sleep(0.2)
        #print(len(t2),'dim')
#        if len(t2)>0:
#            
#            for k in t2:
        distenct=[[(e1 - e2).norm().item() for e2 in embeddings] for e1 in t2]
        
        dist1=np.array(distenct)
        d2=np.argmin(dist1, axis=1)
        
        print(dist1[0][d2[0]])
        if dist1[0][d2[0]]<3:
            print('you are',names[d2[0]])
        #    print(x_aligned, prob)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 存储图片
        cv2.imwrite("camera.jpeg", frame)
        break

cap.release()
cv2.destroyAllWindows()

"""











