'''
    【单个】usb相机的拍摄、每帧处理、保存视频或截图。
'''


import cv2
import numpy as np
import imageio

# 是双目相机
cap = cv2.VideoCapture(1, apiPreference=cv2.CAP_DSHOW)
# videocapture默认读取usb相机的传输格式为YUY2，这种格式会限制opencv读取的视频分辨率和最大帧率，要看相机支持怎样的传输；通常更大的分辨率会导致更小的帧率；
# 要获取更大的分辨率和帧率，必须把格式（fourcc）修改为MJPG，方法为下面第一行，在opencv3中这个设置必须在其他设置之后
# 但是分辨率大了之后帧率就会小（西八

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))

h, w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h, w = int(h), int(w)

# 保存摄像投的数据为视频文件
# video1 = imageio.get_writer('meta/ins/videos/dev2/left.mp4', fps=30)
# video2 = imageio.get_writer('meta/ins/videos/dev2/right.mp4', fps=30)

# 读取内参，去畸变验证
# dev = np.load('result/ins/dev3.npy', allow_pickle=True)

recording = False
idx = 1
while True:
    idx += 1
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cropped1 = frame[0:h, 0:w//2]
    cropped2 = frame[0:h, w//2:w]
    # cropped1 = cv2.flip(cropped1, -1)
    # cropped2 = cv2.flip(cropped2, -1)

    # if recording:
    #     video1.append_data(cropped1)
    #     video2.append_data(cropped2)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    show = np.hstack([cropped1,cropped2])
    cv2.imshow('frame', show)

    # 验证内参
    # M1 = dev.item()['left_mat']
    # D1 = dev.item()['left_dist']
    # M2 = dev.item()['right_mat']
    # D2 = dev.item()['right_dist']

    # new_K1 = np.array([[360,0,640],[0,360,400],[0,0,1]], dtype=np.float32)  # 360 和 640 是像素中心点坐标的真值(图像尺寸1280*720)
    # map1_1, map1_2 = cv2.fisheye.initUndistortRectifyMap(M1, D1, np.eye(3), new_K1, (w//2,h), cv2.CV_16SC2)
    # un1 = cv2.remap(cropped1, map1_1, map1_2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # new_K2 = new_K1.copy()
    # map2_1, map2_2 = cv2.fisheye.initUndistortRectifyMap(M2, D2, np.eye(3), new_K2, (w//2,h), cv2.CV_16SC2)
    # un2 = cv2.remap(cropped2, map2_1, map2_2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # uns = np.hstack([un1,un2])
    # cv2.namedWindow('undistort', cv2.WINDOW_NORMAL)
    # cv2.imshow('undistort', uns)
    
 
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('s') and not recording:
        print('start recording')
        recording = True
    if key & 0xFF == ord('d') and recording:
        print('stop recording')
        recording = False
    if key & 0xFF == ord('z'):
        cv2.imwrite('meta/ins/imgs/dev2/left/raw_' + str(idx) + '.jpg', cropped1)
    if key & 0xFF == ord('x'):
        cv2.imwrite('meta/ins/imgs/dev2/right/raw_' + str(idx) + '.jpg', cropped2)

cap.release()
# video1.close()
# video2.close()
cv2.destroyAllWindows()
