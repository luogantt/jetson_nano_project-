
## Quick start

1. Either install using pip  wait about some minintr:
    ```bash
    pip install  -r requirements.txt
    ```


2. add your face into data/test_image:
    ```
    
    cd ./deep_face/data/test_images
    
    mkdir your_name
    ```
3. add your face picture into /deep_face/data/test_images


4.  run the code :
    ```
    python multi_face_trace_recongnize.py
    ```
    
    
 A full face tracking example by opencv can be found at 

![](data/example/test.png)


在mtcnn 下加入下面代码

if str(boxes)=='None':
	selected_boxes.append(None)
	selected_probs.append([None])
	selected_points.append(None)
	continue
