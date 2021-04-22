#导入包
from hyperlpr import *
#导入OpenCV库
import cv2


#这个函数是设置读取和输出图像高度宽度和帧率
def gstreamer_pipeline(
    capture_width=640,  #宽
    capture_height=360, #高
    display_width=640,
    display_height=360,
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
 


#如果是在jetson nano 上用这个代码，注释掉下一条
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
 
#如果是在 x86 计算平台，比如你的笔记本， 上用这个代码，注释掉上一条
#cap = cv2.VideoCapture(0)

frames = 0


while cap.isOpened():
    ret_val, img = cap.read()  #读取图片
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv reead img as BGR

    # Mirror 
    """
    img = cv2.flip(img, 1)
    img_re = cv2.resize(img, (416, 416))
    """
    #input_imgs = transforms.ToTensor()(img_re)

    result=HyperLPR_plate_recognition(img) #识别出图片中车牌数字
    if len(result)>0:
    	#print(result)
    	if result[0][1]>0.91:  #这个0.91是置信度
    		print(result)  #打印数字
    cv2.imshow('Demo webcam', img)  #显示图片
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 存储图片
        #cv2.imwrite("camera.jpeg", frame)
        break
        
cam.release()
cv2.destroyAllWindows()
