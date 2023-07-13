'''
    可视化多相机位姿。
'''

import cv2
import numpy as np
from camera_pose_visualizer import CameraPoseVisualizer
from scipy.spatial.transform import Rotation as R


def HtoQuatAndT(H):
    r'''
        4*4的外参矩阵 H => 4+3 四元数+位置 的7维向量。
    '''
    Rq = R.from_matrix(H[:3,:3]).as_quat().reshape(1,4)
    T = H[:3,3].reshape(1,3)
    Q = np.hstack((Rq,T))
    return Q  #[1,7]


def constructH(Q):
    r'''
        上面的逆过程。
    '''
    Rm = R.from_quat(Q[:,:4]).as_matrix().squeeze()
    T = Q[:,4:].reshape(3,1)
    rotation_t = np.hstack([Rm, T])
    H = np.vstack([rotation_t,np.array([[0, 0, 0, 1]])])
    return H


def calmeanRandT(path):
    r'''
        通过四元数计算多个相机位姿的均值。只有在所有旋转矩阵都在一定范围内才可以使用这种均值化。
        输入path是npy或npz文件，包括了一个object，存储了四个相机的外参。
    '''
    exdata = np.load(path, allow_pickle=True)
    H2l_2r, H2l_3l, H2l_3r = [],[],[]
    for exs in exdata: 
        H2l = exs['dev2_left']
        H2r = exs['dev2_right']
        H3l = exs['dev3_left']
        H3r = exs['dev3_right']
        if H2l is None or H2r is None or H3l is None or H3r is None:
            continue

        H2r = np.dot(H23r, np.linalg.inv(H23l))
        H3l = np.dot(H34l, np.linalg.inv(H23l))
        H3r = np.dot(H34r, np.linalg.inv(H23l))

        H2l_2r.append(HtoQuatAndT(H2r))
        H2l_3l.append(HtoQuatAndT(H3l))
        H2l_3r.append(HtoQuatAndT(H3r))
    
    Q2r = np.array(H2l_2r).mean(axis=0)
    Q3l = np.array(H2l_3l).mean(axis=0)
    Q3r = np.array(H2l_3r).mean(axis=0)

    return constructH(Q2r), constructH(Q3l), constructH(Q3r)




if __name__ == '__main__':
    visualizer = CameraPoseVisualizer([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])

    exdata = np.load('multiCam_exs.npy', allow_pickle=True) # 替换路径
    H2rs,H3ls,H3rs = [],[],[]
    for exs in exdata: 
        H23l = exs['dev2_left']
        H23r = exs['dev2_right']
        H34l = exs['dev3_left']
        H34r = exs['dev3_right']

        if H23l is None or H23r is None or H34l is None or H34r is None:
            continue

        # 计算相对位姿，全部转移到 H2l坐标系下
        H2r = np.dot(H23l, np.linalg.inv(H23r))
        H3l = np.dot(H23l, np.linalg.inv(H34l))
        H3r = np.dot(H23l, np.linalg.inv(H34r))

        H2rs.append(H2r)
        H3ls.append(H3l)
        H3rs.append(H3r)

        # 标定板可视化
        # visualizer.chessboard()

        # 4相机外参可视化
        # visualizer.extrinsic2pyramid(np.linalg.inv(H23l), 'r', 0.2)
        # visualizer.extrinsic2pyramid(np.linalg.inv(H23r), 'c', 0.2)
        # visualizer.extrinsic2pyramid(np.linalg.inv(H34l), 'k', 0.2)
        # visualizer.extrinsic2pyramid(np.linalg.inv(H34r), 'b', 0.2)

        visualizer.extrinsic2pyramid(np.eye(4), 'r', 0.2)
        visualizer.extrinsic2pyramid(H2r, 'c', 0.2)
        visualizer.extrinsic2pyramid(H3l, 'k', 0.2)
        visualizer.extrinsic2pyramid(H3r, 'b', 0.2)
    
    # meanH2r, meanH3l, meanH3r = calmeanRandT('stereos/extrinsic_results/dev2and3_o.npy')
    # visualizer.extrinsic2pyramid(meanH2r, 'c', 0.4)
    # visualizer.extrinsic2pyramid(meanH3l, 'k', 0.4)
    # visualizer.extrinsic2pyramid(meanH3r, 'b', 0.4)

    visualizer.show()
