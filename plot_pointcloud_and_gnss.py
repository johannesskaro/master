import numpy as np
from scipy.spatial.transform import Rotation as R

import pyzed.sl as sl

from extrinsics import H_POINTS_FLOOR_FROM_LIDAR, H_POINTS_LEFT_CAM_FROM_FLOOR, H_POINTS_LEFT_ZED_FROM_LEFT_CAM, H_POINTS_PIREN_ENU_FROM_PIREN, H_POINTS_VESSEL_FROM_FLOOR, invert_transformation

import sys
sys.path.insert(0, "/home/nicholas/GitHub/phd-stereo/python_tools")
from o3d_pc_visualizer import O3DPointCloudVisualizer
from stereo_svo import SVOCamera
from data_loading_2023e import gen_lidar_pc, get_all_ma2_pose_ned, get_kb2_gnss_enu, get_nt_gnss_enu, homogeneous_multiplication

ROSBAG_FOLDER = "/home/nicholas/BigData/2023 Summer experiment MA2 rosbags"
GNSS_TOPIC = "/senti_parser/SentiPose"
LIDAR_TOPIC = "/lidar_aft/points"
 

ROSBAG_NAME = "scen6" # Docking 2
SVO_FILE_PATH = "/media/nicholas/T7 Shield/2023-07-11 Multi ZED Summer/ZED camera svo files/2023-07-11_12-55-58_5256916_HD1080_FPS15.svo" # Docking 2, left
GNSS_MINUS_ZED_TIME_NS = -201327414.75 # Docing 2
KB2_FILE_PATH = "/media/nicholas/T7 Shield/2023-07-11 Multi ZED Summer/KBox2/LOG00055.pos" # Docking 1 and 2
NT_FILE_PATH = "/media/nicholas/T7 Shield/2023-07-11 Multi ZED Summer/NTBox/gnssdump-20230710171814.pos" # Docking 1 and 2
ZED_START_TIME = 1689073011222142269 # Docking 2

# ZED_START_TIME = 1689068851366919729 # Buster maneuver
# ROSBAG_NAME = "scen3" # Buster maneuver
# SVO_FILE_PATH = "/media/nicholas/T7 Shield/2023-07-11 Multi ZED Summer/ZED camera svo files/2023-07-11_11-46-15_5256916_HD1080_FPS15.svo" # Buster maneuver.
# NT_FILE_PATH = "/media/nicholas/T7 Shield/2023-07-11 Multi ZED Summer/NTBox/gnssdump-20230710170149.pos" # Buster maneuver
# KB2_FILE_PATH = "/media/nicholas/T7 Shield/2023-07-11 Multi ZED Summer/KBox2/LOG00051.pos" # Buster maneuver
# GNSS_MINUS_ZED_TIME_NS = -136682008.0 # Buster maneuver

class DataPlayer():
    def __init__(self) -> None:
        self.ma2_t_pos_ori_ned = get_all_ma2_pose_ned(f"{ROSBAG_FOLDER}/{ROSBAG_NAME}")
        self.stereo_cam = SVOCamera(SVO_FILE_PATH)
        # self.stereo_cam.set_svo_position(0)
        self.current_time =  ZED_START_TIME
        self.stereo_cam.set_svo_position_timestamp(self.current_time)
        self.kb2_enu = get_kb2_gnss_enu(KB2_FILE_PATH)
        self.nt_enu = get_nt_gnss_enu(NT_FILE_PATH)
        self.lidar_pcs_gen = gen_lidar_pc(f"{ROSBAG_FOLDER}/{ROSBAG_NAME}")
        self.last_lidar_t_xyz_rgb = next(self.lidar_pcs_gen)

    def grab(self): # data_player.grab() == sl.ERROR_CODE.SUCCESS
        # self.current_time += 0.5*10**9
        # self.stereo_cam.set_svo_position_timestamp(self.current_time)
        return self.stereo_cam.grab()
    
    def get_point_cloud(self):
        pc_xyz_zed, pc_rgb = self.stereo_cam.get_neural_numpy_pointcloud()
        H_points_enu_from_zed = self._get_transform_enu_from_zed()
        pc_xyz_enu = homogeneous_multiplication(H_points_enu_from_zed, pc_xyz_zed)
        return pc_xyz_enu, pc_rgb
    
    def _get_transform_enu_from_zed(self):
        H_points_piren_from_vessel = self._get_transform_piren_ned_from_vessel()
        H_POINTS_FLOOR_FROM_LEFT_CAM = invert_transformation(H_POINTS_LEFT_CAM_FROM_FLOOR)
        H_POINTS_LEFT_CAM_FROM_LEFT_ZED = invert_transformation(H_POINTS_LEFT_ZED_FROM_LEFT_CAM)
        H_points_enu_from_zed = H_POINTS_PIREN_ENU_FROM_PIREN @ H_points_piren_from_vessel @ H_POINTS_VESSEL_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LEFT_CAM @ H_POINTS_LEFT_CAM_FROM_LEFT_ZED
        return H_points_enu_from_zed

    def _get_transform_piren_ned_from_vessel(self):
        ma2_tpose = self._get_ma2_tpose_ned()
        ma2_trans = ma2_tpose[1:4]
        ma2_rot_quat = ma2_tpose[4:]
        ma2_rot_mat = R.from_quat(ma2_rot_quat).as_matrix()

        H_points_piren_from_vessel = np.block([
            [ma2_rot_mat, ma2_trans[:,np.newaxis]], 
            [np.zeros((1,3)), np.ones((1,1))]
        ])
        return H_points_piren_from_vessel
    
    def _get_ma2_tpose_ned(self):
        timestamp = self.get_timestamp()
        ma2_ned = self._get_row_from_timestamp(timestamp, self.ma2_t_pos_ori_ned)
        return ma2_ned
    
    def get_ma2_pose_enu(self):
        H_points_piren_from_vessel = self._get_transform_piren_ned_from_vessel()
        H_points_enu_from_vessel = H_POINTS_PIREN_ENU_FROM_PIREN @ H_points_piren_from_vessel
        R_enu = H_points_enu_from_vessel[0:3, 0:3]
        rot = R.from_matrix(R_enu)
        rot_quat = rot.as_quat()
        t_enu = H_points_enu_from_vessel[0:3, 3]
        return t_enu, rot_quat

    def get_zed_pose_enu(self):
        H_points_enu_from_zed = self._get_transform_enu_from_zed()

        H = H_points_enu_from_zed
        R_enu = H[0:3, 0:3]
        rot = R.from_matrix(R_enu)
        rot_quat = rot.as_quat()
        t_enu = H[0:3, 3]
        return t_enu, rot_quat

    def get_timestamp(self):
        return self.stereo_cam.get_timestamp() + GNSS_MINUS_ZED_TIME_NS
    
    def _get_row_from_timestamp(self, timestamp, table):
        idx = np.searchsorted(table[:,0], timestamp, "left")
        if idx == 0: 
            print(f"Timestamp before start of table. ")
            return table[idx]
        if idx == table.shape[0]: 
            print(f"Timestamp after end of table. ")
            return table[-1]
        if abs(table[idx-1][0] - timestamp) < abs(table[idx][0] - timestamp):
            return table[idx-1]
        else:
            return table[idx]
    
    def get_kb2_pos_enu(self):
        timestamp = self.get_timestamp()
        kb2_pos_row = self._get_row_from_timestamp(timestamp, self.kb2_enu)
        kb2_pos = kb2_pos_row[1:] + np.array([0, 0, 6.5])
        return kb2_pos
    
    def get_nt_pos_enu(self):
        timestamp = self.get_timestamp()
        nt_pos_row = self._get_row_from_timestamp(timestamp, self.nt_enu)
        nt_pos = nt_pos_row[1:] + np.array([0, 0, 8])
        return nt_pos
    
    def get_lidar_point_cloud(self):
        H_points_piren_from_vessel = self._get_transform_piren_ned_from_vessel()
        H_points_enu_from_lidar = H_POINTS_PIREN_ENU_FROM_PIREN @ H_points_piren_from_vessel @ H_POINTS_VESSEL_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LIDAR

        pc_xyz, pc_rgb = self._get_lidar_point_cloud_lidar_frame()
        pc_xyz_enu = homogeneous_multiplication(H_points_enu_from_lidar, pc_xyz)

        return pc_xyz_enu, pc_rgb
    
    def _get_lidar_point_cloud_lidar_frame(self):
        timestamp = self.get_timestamp()

        while self.last_lidar_t_xyz_rgb[0] < timestamp:
            try:
                self.last_lidar_t_xyz_rgb = next(self.lidar_pcs_gen)
            except StopIteration:
                print("No more lidar point clouds available. ")
                return self.last_lidar_t_xyz_rgb[1], self.last_lidar_t_xyz_rgb[2]
        return self.last_lidar_t_xyz_rgb[1], self.last_lidar_t_xyz_rgb[2]

def main():
    data_player = DataPlayer()

    o3d_vis = O3DPointCloudVisualizer(
        visualization_parameter_path=f"{ROSBAG_FOLDER}/o3d_vis.json"
    )
    o3d_pc = o3d_vis.create_point_cloud()
    o3d_ma2_coord_frame = o3d_vis.create_coord_frame()
    o3d_zed_coord_frame = o3d_vis.create_coord_frame()
    o3d_kb2_point = o3d_vis.create_point()
    o3d_nt_point = o3d_vis.create_point()
    o3d_lidar_pc = o3d_vis.create_point_cloud()

    while data_player.grab() == sl.ERROR_CODE.SUCCESS:
        pc_xyz, pc_rgb = data_player.get_point_cloud()
        ma2_pos, ma2_quat = data_player.get_ma2_pose_enu()
        zed_pos, zed_quat = data_player.get_zed_pose_enu()
        kb2_pos = data_player.get_kb2_pos_enu()
        nt_pos = data_player.get_nt_pos_enu()
        lidar_xyz, lidar_rgb = data_player.get_lidar_point_cloud()

        o3d_vis.update_point_cloud(o3d_lidar_pc, lidar_xyz, lidar_rgb)
        o3d_vis.update_point(o3d_nt_point, nt_pos)
        o3d_vis.update_point(o3d_kb2_point, kb2_pos)
        o3d_vis.update_coord_frame(o3d_zed_coord_frame, zed_pos, zed_quat)
        o3d_vis.update_coord_frame(o3d_ma2_coord_frame, ma2_pos, ma2_quat)
        o3d_vis.update_point_cloud(o3d_pc, pc_xyz, pc_rgb)
        o3d_vis.render()

if __name__ == "__main__":
    main()