import numpy as np
from scipy.spatial.transform import Rotation as R

OFFSET_LAT = - 0.45e-04   # 1.2091700000382843e-05 # - 0.45e-04
OFFSET_LON = 2.0e-04    #-2.2780000000111045e-05 # 2.0e-04
PIREN_LAT = 63.4389029083 + OFFSET_LAT
PIREN_LON = 10.39908278 + OFFSET_LON
PIREN_ALT = 39.923

ROT_FLOOR_TO_LIDAR = np.array([[-8.27535228e-01,  5.61392452e-01,  4.89505779e-03],
       [ 5.61413072e-01,  8.27516685e-01,  5.61236993e-03],
       [-8.99999879e-04,  7.39258326e-03, -9.99972269e-01]])
TRANS_FLOOR_TO_LIDAR = np.array([-4.1091, -1.1602, -1.015 ])
H_POINTS_FLOOR_FROM_LIDAR = np.block([
    [ROT_FLOOR_TO_LIDAR, TRANS_FLOOR_TO_LIDAR[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])

# Using left MA2 cam pose
ROT_FLOOR_TO_CAM = np.array([[-3.20510335e-09, -3.20510335e-09, -1.00000000e+00],
       [-1.00000000e+00, -5.55111512e-17,  3.20510335e-09],
       [-5.55111512e-17,  1.00000000e+00, -3.20510335e-09]])
TRANS_FLOOR_TO_CAM = np.array([-4.1358,  1.0967, -0.702 ])

H_POINTS_CAM_FROM_FLOOR = np.block([
    [ROT_FLOOR_TO_CAM.T, -ROT_FLOOR_TO_CAM.T.dot(TRANS_FLOOR_TO_CAM)[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])

ROT_EXTRINSIC = np.array([[ 0.99940448, -0.0263474 ,  0.02228227],
                          [ 0.02664859,  0.99955599, -0.01332975],
                          [-0.02192117,  0.01391561,  0.99966285]])
TRANS_EXTRINSIC = np.array([[-1.95],
                            [-0.02335371],
                            [ 0.2]])
ROT_EXTRINSIC = R.from_euler("y", [-2], degrees=True).as_matrix()[0] @ ROT_EXTRINSIC
H_POINTS_RIGHT_FROM_LEFT_ZED = np.block([
    [ROT_EXTRINSIC, TRANS_EXTRINSIC], 
    [np.zeros((1,3)), np.ones((1,1))]
])

H_POINTS_LEFT_ZED_FROM_LEFT_CAM = np.array([
    [ 0.99987632,  0.00293081,  0.01545154, -0.04023359],
    [-0.00137582,  0.99501634, -0.09970255,  0.353144  ],
    [-0.01566675,  0.09966896,  0.99489731, -0.04098881],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
ROT_FLOOR_TO_LEFT_CAM = np.array([[-3.20510335e-09, -3.20510335e-09, -1.00000000e+00],
       [-1.00000000e+00, -5.55111512e-17,  3.20510335e-09],
       [-5.55111512e-17,  1.00000000e+00, -3.20510335e-09]])
TRANS_FLOOR_TO_LEFT_CAM = np.array([-4.1358,  1.0967, -0.702 ])

H_POINTS_LEFT_CAM_FROM_FLOOR = np.block([
    [ROT_FLOOR_TO_LEFT_CAM.T, -ROT_FLOOR_TO_LEFT_CAM.T.dot(TRANS_FLOOR_TO_LEFT_CAM)[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])

ROT_PIREN_TO_PIREN_ENU = np.array([[ 4.89658314e-12,  1.00000000e+00, -2.06823107e-13],
       [ 1.00000000e+00, -4.89658314e-12,  1.01273332e-24],
       [-1.26217745e-29, -2.06823107e-13, -1.00000000e+00]])
TRANS_PIREN_TO_PIREN_ENU = np.array([0., 0., 0.])
H_POINTS_PIREN_FROM_PIREN_ENU = np.block([
    [ROT_PIREN_TO_PIREN_ENU, TRANS_PIREN_TO_PIREN_ENU[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])
H_POINTS_PIREN_ENU_FROM_PIREN = np.linalg.inv(H_POINTS_PIREN_FROM_PIREN_ENU)

H_POINTS_LEFT_ZED_FROM_LIDAR = H_POINTS_LEFT_ZED_FROM_LEFT_CAM @ H_POINTS_LEFT_CAM_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LIDAR
#H = H_POINTS_LEFT_ZED_FROM_LIDAR  #Use left zed
H = H_POINTS_RIGHT_FROM_LEFT_ZED @ H_POINTS_LEFT_ZED_FROM_LEFT_CAM @ H_POINTS_CAM_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LIDAR #Use right zed
H_INV_TO_RIGHT_ZED = np.linalg.inv(H_POINTS_LEFT_ZED_FROM_LEFT_CAM @ H_POINTS_CAM_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LIDAR)


ROT_VESSEL_TO_FLOOR = np.array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
TRANS_VESSEL_TO_FLOOR = np.array([ 0. ,  0. , -0.3])
H_POINTS_VESSEL_FROM_FLOOR = np.block([
    [ROT_VESSEL_TO_FLOOR, TRANS_VESSEL_TO_FLOOR[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])