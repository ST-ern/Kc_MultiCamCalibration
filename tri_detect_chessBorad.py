'''
    基于opencv的三角测量，计算 估计出的角点距离 和 实际棋盘格边长 的差异。
'''

import cv2
import numpy as np
import sys
# sys.path.append('./utils')
from utils.camera_pose_visualizer import CameraPoseVisualizer


def multitrangleDetect(Hs, points, res=None):
    # points: [n*m*2]， Hs:[m*4*4]; 其中n是目标点的数量，m是对应相机的个数
    Hs = np.array(Hs)
    pos = []
    for p in points:
        matA = []
        for i in range(Hs.shape[0]):
            if p[i][0] is None or p[i][1] is None:
                continue
            x = p[i][0]
            y = p[i][1]
            H = Hs[i]
            (R11,R12,R13,T1) = H[0]
            (R21,R22,R23,T2) = H[1]
            (R31,R32,R33,T3) = H[2]

            line1 = np.array([x*R31-R11, x*R32-R12, x*R33-R13, x*T3-T1])
            line2 = np.array([y*R31-R21, y*R32-R22, y*R33-R23, y*T3-T2])
            matA.append(line1)
            matA.append(line2)
        matA = np.array(matA)   # [m,4]
        u, sigma, vt = np.linalg.svd(matA)  # 这里的vt是V矩阵的转置的意思，最优解变成了最后一行
        pos.append(vt[-1])
    return np.array(pos).T


def triangleuateDetectBoard(M1,D1,H1,M2,D2,H2, points1, points2, M3,D3,H3,M4,D4,H4, points3, points4):   
    points1 = points1.reshape(-1,1,2)
    points2 = points2.reshape(-1,1,2)
    undist1 = cv2.fisheye.undistortPoints(points1, M1, D1)
    undist2 = cv2.fisheye.undistortPoints(points2, M2, D2)
    points3 = points3.reshape(-1,1,2)
    points4 = points4.reshape(-1,1,2)
    undist3 = cv2.fisheye.undistortPoints(points3, M3, D3)
    undist4 = cv2.fisheye.undistortPoints(points4, M4, D4)

    Rt1 = H1[:3]
    Rt2 = H2[:3]
    Rt3 = H3[:3]
    Rt4 = H4[:3]
    Rt1_ = np.dot(H1, np.linalg.inv(H1))[:3]
    Rt2_ = np.dot(H2, np.linalg.inv(H1))[:3]

    res = cv2.triangulatePoints(Rt1, Rt2, undist1, undist2)
    res_ = cv2.triangulatePoints(Rt1_, Rt2_, undist1, undist2)

    # res_tmp = multitrangleDetect([H1,H2], np.concatenate((undist1, undist2), axis=1), res)
    res_tmp2 = multitrangleDetect([H1,H2,H3,H4], np.concatenate((undist1, undist2, undist3, undist4), axis=1), res)
    return res, res_, res_tmp2


if __name__ == '__main__':
    ins_path2 = 'results/ins/dev2_FINAL.npy'
    ins_path3 = 'results/ins/dev3_FINAL.npy'

    dev2 = np.load(ins_path2, allow_pickle=True)
    M1 = dev2.item()['left_mat']
    D1 = dev2.item()['left_dist']
    M2 = dev2.item()['right_mat']
    D2 = dev2.item()['right_dist']
    dev3 = np.load(ins_path2, allow_pickle=True)
    M3 = dev3.item()['left_mat']
    D3 = dev3.item()['left_dist']
    M4 = dev3.item()['right_mat']
    D4 = dev3.item()['right_dist']

    exdata = np.load('result/exs/dev2and3.npy', allow_pickle=True)[19]
    H1 = exdata['dev2_left']
    H2 = exdata['dev2_right']
    H3 = exdata['dev3_left']
    H4 = exdata['dev3_right']

    img1 = 'meta/exs/dev2/left/raw_85.jpg'
    img2 = 'meta/exs/dev2/right/raw_85.jpg'
    img3 = 'meta/exs/dev3/left/raw_85.jpg'
    img4 = 'meta/exs/dev3/right/raw_85.jpg'
    image1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2GRAY)
    image3 = cv2.cvtColor(cv2.imread(img3), cv2.COLOR_BGR2GRAY)
    image4 = cv2.cvtColor(cv2.imread(img4), cv2.COLOR_BGR2GRAY)
    (h, w) = (10, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ret1, imgp_left2 = cv2.findChessboardCornersSB(image1, (h, w), flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    ret2, imgp_right2 = cv2.findChessboardCornersSB(image2, (h, w), flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    ret3, imgp_left3 = cv2.findChessboardCornersSB(image3, (h, w), flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    ret4, imgp_right3 = cv2.findChessboardCornersSB(image4, (h, w), flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    if not ret1 or not ret2 or not ret3 or not ret4:
        print('fail to detect')
    else:
        imgp_left2 = cv2.cornerSubPix(image1, imgp_left2, (5, 5), (-1, -1), criteria)
        imgp_right2 = cv2.cornerSubPix(image2, imgp_right2, (5, 5), (-1, -1), criteria)
        imgp_left3 = cv2.cornerSubPix(image3, imgp_left3, (5, 5), (-1, -1), criteria)
        imgp_right3 = cv2.cornerSubPix(image4, imgp_right3, (5, 5), (-1, -1), criteria)

        point4D, point4D_, point_allCam = triangleuateDetectBoard(M1,D1,H1,M2,D2,H2,imgp_left2, imgp_right2, M3,D3,H3,M4,D4,H4,imgp_left3, imgp_right3)
        point4D, point4D_, point_allCam = np.array(point4D).T, np.array(point4D_).T, np.array(point_allCam).T
        for i in range(point4D.shape[0]):
            point4D[i] = point4D[i] / point4D[i][3]
        for i in range(point4D_.shape[0]):
            point4D_[i] = point4D_[i] / point4D_[i][3]
        for i in range(point_allCam.shape[0]):
            point_allCam[i] = point_allCam[i] / point_allCam[i][3]

        point4D_2H1 = np.dot(H1, point4D.T).T
        
        # 测距验证-----
        # 重新计算了三维点的距离，误差在1.5mm以下(2个相机)
        # 三维点的距离误差到了0.7mm以下（4个相机）
        pos_w = point4D_[:,:3]
        dif = pos_w[1:] - pos_w[:-1]
        dist = np.linalg.norm(np.array(dif), axis=1)    

        visualizer = CameraPoseVisualizer([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])
        visualizer.chessboard(wp=pos_w)
        # visualizer.chessboard(wp=pos_w_)
        # visualizer.extrinsic2pyramid(np.linalg.inv(H1), 'r', 0.2)
        # visualizer.extrinsic2pyramid(np.linalg.inv(H2), 'c', 0.2)
        # visualizer.extrinsic2pyramid(np.linalg.inv(H3), 'k', 0.2)
        # visualizer.extrinsic2pyramid(np.linalg.inv(H4), 'b', 0.2)
        visualizer.extrinsic2pyramid(np.dot(H1, np.linalg.inv(H1)), 'r', 0.2)
        visualizer.extrinsic2pyramid(np.dot(H1, np.linalg.inv(H2)), 'c', 0.2)
        visualizer.extrinsic2pyramid(np.dot(H1, np.linalg.inv(H3)), 'k', 0.2)
        visualizer.extrinsic2pyramid(np.dot(H1, np.linalg.inv(H4)), 'b', 0.2)

        visualizer.show()
