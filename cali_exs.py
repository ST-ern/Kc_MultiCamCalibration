'''
    相机外参计算。基于opencv。
'''


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def cali_cam_ex_solvePnP(ins_path, img_left, img_right):
    r'''
        计算两个相机相对标定板的外参。
    '''
    dev = np.load(ins_path, allow_pickle=True)
    M1 = dev.item()['left_mat']
    D1 = dev.item()['left_dist']
    M2 = dev.item()['right_mat']
    D2 = dev.item()['right_dist']

    image_l = cv2.imread(img_left)
    gray_l = cv2.cvtColor(image_l, cv2.COLOR_RGB2GRAY)
    H1, R1, T1 = calibration_cam_ex_fromImg_solvePnP(gray_l, M1, D1)

    image_r = cv2.imread(img_right)
    gray_r = cv2.cvtColor(image_r, cv2.COLOR_RGB2GRAY)
    H2, R2, T2 = calibration_cam_ex_fromImg_solvePnP(gray_r, M2, D2)

    return H1, H2, R1,R2, T1,T2


def calibration_cam_ex_fromImg_solvePnP(gray, mtx, dist):
    r'''
        外参标定。
    '''
    pattern_size = (10, 7)  # 自己设置
    h, w = pattern_size

    # 设置(生成)标定图在世界坐标中的坐标
    world_point = np.zeros((h * w, 3), np.float32)
    world_point[:, :2] = 0.025 * np.mgrid[:h, :w].T.reshape(-1, 2)  # 0.025是实际的棋盘格长度
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ret, corners =  cv2.findChessboardCornersSB(gray, (h, w), flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    # draw = cv2.drawChessboardCorners(gray, (h,w), corners, ret)
    if ret:
        exact_corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria) # [N,1,2]

        # 针孔相机可以不需要在这里去畸变，改solvePnP的内参就好
        distort_corners = cv2.fisheye.undistortPoints(exact_corners, mtx, dist, None, mtx)
        no_dist = np.array([0,0,0,0], dtype=np.float32).reshape(4,1)
        flags = cv2.SOLVEPNP_ITERATIVE

        # 获取外参
        # _, rvec, tvec = cv2.solvePnP(world_point, exact_corners, mtx, dist)             # 没先去畸变
        # retval, rvec, tvec, linears = cv2.solvePnPRansac(world_point, distort_corners, mtx, no_dist)   # solvePnPRansac和solvePnP没区别
        retval, rvec, tvec = cv2.solvePnP(world_point, distort_corners, mtx, no_dist, flags=flags)     # 先鱼眼去畸变
        # 获得的旋转矩阵是向量，是3×1的矩阵，想要还原回3×3的矩阵，需要罗德里格斯变换Rodrigues，
        rotation_m, _ = cv2.Rodrigues(rvec) #罗德里格斯变换
        # print(rotation_m)
        # print('旋转矩阵：', rvec)
        # print('平移矩阵: ', tvec)

        rotation_t = np.hstack([rotation_m, tvec])
        rotation_t_Homogeneous_matrix = np.vstack([rotation_t,np.array([[0, 0, 0, 1]])])
 
        # 额外看看重投影误差
        img_point_remap, jaco = cv2.fisheye.projectPoints(world_point.reshape(-1,1,3), rvec, tvec, mtx, dist)
        remap_dif = np.linalg.norm(exact_corners-img_point_remap)
        # print(remap_dif / exact_corners.shape[0])

        return rotation_t_Homogeneous_matrix, rotation_m, tvec
    else:
        return None, None, None
    


def calibrate_cam_ex_fromImg_stereo(folder1, folder2, cam_in1, dist1, cam_in2, dist2):
    r'''
        双目标定。（？）
    '''
    pattern_size = (10, 7)
    h, w = pattern_size

    K_left = np.array(cam_in1)
    D_left = np.array(dist1)
    K_right = np.array(cam_in2)
    D_right = np.array(dist2)

    R = np.zeros((1, 1, 3), dtype=np.float64)
    T = np.zeros((1, 1, 3), dtype=np.float64)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    imgpoints_left, imgpoints_right = [],[]
        

    for file_name in os.listdir(folder1):
        path1 = folder1 + '/' + file_name
        path2 = folder2 + '/' + file_name

        image1 = cv2.imread(path1)
        image2 = cv2.imread(path2)

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        ret1, imgp_left = cv2.findChessboardCornersSB(image1, (h, w), flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        ret2, imgp_right = cv2.findChessboardCornersSB(image2, (h, w), flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        if ret1 and ret2:
            imgp_left = cv2.cornerSubPix(image1, imgp_left, (5, 5), (-1, -1), criteria)
            imgp_right = cv2.cornerSubPix(image2, imgp_right, (5, 5), (-1, -1), criteria)

            imgpoints_left.append(imgp_left)
            imgpoints_right.append(imgp_right)
        else:
            continue
    
    objp = np.zeros((1, h*w, 3), np.float32)
    objp[0,:,:2] = 0.025 * np.mgrid[0:h, 0:w].T.reshape(-1, 2)

    N_ok = len(imgpoints_left)
    objpoints = np.asarray([objp] * N_ok, dtype=np.float64)
    imgpoints_left = np.asarray(imgpoints_left, dtype=np.float64)
    imgpoints_right = np.asarray(imgpoints_right, dtype=np.float64)
    objpoints = np.reshape(objpoints, (N_ok, 1, h*w, 3))
    imgpoints_left = np.reshape(imgpoints_left, (N_ok, 1, h*w, 2))
    imgpoints_right = np.reshape(imgpoints_right, (N_ok, 1, h*w, 2))
    # objpoints shape: (  <  num of calibration images>, 1,  < num points in set>, 3)
    # imgpoints_left shape: ( < num of calibration images>, 1,  < num points in set>, 2)
    # imgpoints_right shape: ( < num of calibration images>, 1,  < num points in set>, 2) 都先把images num去掉
    # print(objpoints.shape)
    # print(imgpoints_left.shape)
    # print(imgpoints_right.shape)

    rms, new_K_left, new_D_left, new_K_right, new_D_right, new_R, new_T = cv2.fisheye.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, K_left, D_left, K_right, D_right, image1.shape, R, T,
        cv2.fisheye.CALIB_FIX_SKEW+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_INTRINSIC, criteria,
    )
    return new_R, new_T



if __name__ == '__main__':
    dev2_in_path = 'results/ins/dev2_FINAL.npy'
    dev3_in_path = 'results/ins/dev3_FINAL.npy'

    dev2_l_imgs = 'meta/exs/dev2/left'
    dev2_r_imgs = 'meta/exs/dev2/right'
    dev3_l_imgs = 'meta/exs/dev3/left'
    dev3_r_imgs = 'meta/exs/dev3/right'

    exs = []
    names = []

    # 计算外参 -- solvePnP -------------------------------------------------
    for filename in os.listdir(dev2_l_imgs):
        dev2_l_img = dev2_l_imgs + '/' + filename
        dev2_r_img = dev2_r_imgs + '/' + filename
        dev3_l_img = dev3_l_imgs + '/' + filename
        dev3_r_img = dev3_r_imgs + '/' + filename
    
        H2l, H2r, R2l,R2r, T2l,T2r = cali_cam_ex_solvePnP(dev2_in_path, dev2_l_img, dev2_r_img)
        H3l, H3r, R3l,R3r, T3l,T3r = cali_cam_ex_solvePnP(dev3_in_path, dev3_l_img, dev3_r_img)

        ex = {'dev2_left': H2l, 'dev2_right': H2r, 'dev3_left': H3l, 'dev3_right': H3r}
        print(exs)
        exs.append(ex)
        names.append(filename)
        # if H23l is not None and H23r is not None:
        #     H2r_l = np.dot(H23r, np.linalg.inv(H23l))
        #     H2r_ls.append(H2r_l)
    np.save('results/exs/dev2and3.npy', exs)


    # 计算外参 -- stereo -----------------------------------------------
    # dev = np.load(dev2_in_path, allow_pickle=True)
    # M1 = dev.item()['left_mat']
    # D1 = dev.item()['left_dist']
    # M2 = dev.item()['right_mat']
    # D2 = dev.item()['right_dist']
    # R1, T1 = calibrate_cam_ex_fromImg_stereo(dev2_l_imgs, dev2_r_imgs, M1,D1,M2,D2)
    # print(R1)
