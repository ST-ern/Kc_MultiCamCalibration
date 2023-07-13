'''
    自己实现的多目测量重建，计算 估计出的关键点 在第一相机坐标系内的坐标。
'''


import cv2
import numpy as np
from tri_detect_chessBorad import multitrangleDetect
import sys
# sys.path.append('./')
from utils.camera_pose_visualizer import CameraPoseVisualizer


def triangleuateDetectPoints(Ms, Ds, Hs, distPoints):
    # distPoints:[n,m,2];n:相机数量,m:关键点数量
    distPoints = np.array(distPoints, dtype=np.float32).transpose(0,2,1)
    undists = []
    for i in range(distPoints.shape[0]):
        undist = cv2.fisheye.undistortPoints(distPoints[i].reshape(-1,1,2), Ms[i], Ds[i]).transpose(1,0,2)   # [1,21(m),2]
        undists.append(undist.squeeze(0))
    undists = np.array(undists).transpose(1,0,2)    # [m,n,2]
    res = multitrangleDetect(Hs, undists)

    res = np.array(res).T
    for i in range(res.shape[0]):
        res[i] = res[i] / res[i][3]
    return res[:,:3]


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

    exdata = np.load('results/exs/dev2and3.npy', allow_pickle=True)[19]
    H1 = exdata['dev2_left']
    H2 = exdata['dev2_right']
    H3 = exdata['dev3_left']
    H4 = exdata['dev3_right']


    point1 = np.load('meta/hands/dev2/left_h_190.npy')
    point2 = np.load('meta/hands/dev2/right_h_190.npy')
    point3 = np.load('meta/hands/dev3/left_h_190.npy')
    point4 = np.load('meta/hands/dev3/right_h_190.npy')

    p4D = triangleuateDetectPoints([M1,M2,M3,M4], [D1,D2,D3,D4], 
                                   [np.dot(H1, np.linalg.inv(H1)),np.dot(H2, np.linalg.inv(H1)),np.dot(H3, np.linalg.inv(H1)),np.dot(H4, np.linalg.inv(H1))],
                                   [point1,point2,point3,point4] )  # [n,4]
    
    visualizer = CameraPoseVisualizer([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])
    xs = p4D[:,0].T
    ys = p4D[:,1].T
    zs = p4D[:,2].T
    visualizer.ax.scatter(xs, ys, zs, s=5, c='g')

    visualizer.extrinsic2pyramid(np.dot(H1, np.linalg.inv(H1)), 'r', 0.2)
    visualizer.extrinsic2pyramid(np.dot(H1, np.linalg.inv(H2)), 'c', 0.2)
    visualizer.extrinsic2pyramid(np.dot(H1, np.linalg.inv(H3)), 'k', 0.2)
    visualizer.extrinsic2pyramid(np.dot(H1, np.linalg.inv(H4)), 'b', 0.2)
    visualizer.show()


