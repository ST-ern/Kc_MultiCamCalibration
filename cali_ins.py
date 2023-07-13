'''
    内参标定。基于opencv。
'''

import cv2
import os
import yaml
import json
import numpy as np



def cali_single(img_folder):
    pattern_size = (10, 7)  # 根据实际情况修改
    h, w = pattern_size
    objp = np.zeros((1, h*w, 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space -> 是固定的XY平面里的坐标，Z=0是为了简便计算（物点）
    imgpoints = []  # 2d points in image plane.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    idx = 2
    names_list = []
    
    for filename in os.listdir(img_folder):
        idx += 1
        if idx % 1 == 0:    # 从视频读图，图太多了，采样一下

            img = cv2.imread(img_folder + "/" + filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCornersSB(gray, (h, w), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                # print("find ", idx)
                corners = cv2.cornerSubPix(
                    gray, corners, (5, 5), (-1, -1), criteria
                )  # finetune window size to refine corners -》 精确确定角的位置（2d）

                objpoints.append(objp)      # 3d物点（角点的物点）  -> Z为0的3d点【因为坐标系可以自定义？】
                imgpoints.append(corners)   # 2d图像点             -> cv2.findChessboardCorners + cv2.cornerSubPix
                names_list.append(filename)

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    calibration_flags = cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND 

    # 如果有CALIB_CHECK_COND问题看namelists删图片
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, gray.shape[::-1], K, D, rvecs, tvecs, calibration_flags, criteria
    )   # 鱼眼相机
    # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    #     objpoints, imgpoints, gray.shape[::-1], None, None
    # )     # 针孔相机
    print(names_list[1])
    print(cv2.Rodrigues(rvecs[1])[0])
    print(tvecs[1])
    return ret, camera_matrix, dist_coeffs



if __name__ == '__main__':
    ret1, M1, D1 = cali_single('meta/ins/imgs/dev2/left')
    # ret2, M2, D2 = cali_single('meta/ins/imgs/dev2/right')
    # dev = {'left_mat': M1, 'left_dist': D1, 'right_mat': M2, 'right_dist': D2}
    # np.save('results/ins/dev2_fisheye.npy', dev)
    print(ret1)
    # print(ret2)



    # 去畸变验证
    # cropped1 = cv2.imread('meta/ins/imgs/dev2/left/61.jpg')
    # dev = np.load('results/ins/dev2_FINAL.npy', allow_pickle=True)
    # M1 = dev.item()['left_mat']
    # D1 = dev.item()['left_dist']
    # new_K1 = np.array([[360,0,640],[0,360,400],[0,0,1]], dtype=np.float32)
    # map1_1, map1_2 = cv2.fisheye.initUndistortRectifyMap(M1, D1, np.eye(3), new_K1, (1280,800), cv2.CV_16SC2)
    # un1 = cv2.remap(cropped1, map1_1, map1_2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # cv2.imwrite('stereos/instrinsic_videoandimg/imgs/dev2/61_fisheye.jpg', un1)

