import json
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import pymap3d
from transforms import *
import cv2


def transform_to_utm32(lon, lat):
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

ORIGIN = transform_to_utm32(PIREN_LON, PIREN_LAT)

def get_line_strings_from_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    # Convert back to LineString objects
    line_strings = [LineString(coords) for coords in data]
    return line_strings

LINE_STRINGS = get_line_strings_from_file("files/linestrings.json")

def plot_line_strings(ax, line_strings, origin=[0, 0]):
    origin_x = origin[0]
    origin_y = origin[1]
    

    plt.figure(figsize=(8, 8))
    for ls in line_strings:
        x, y = ls.xy
        x = [x_ - origin_x for x_ in x]
        y = [y_ - origin_y for y_ in y]
        ax.plot(x, y, color='black', linestyle='-')

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("UTM32 LineStrings")
    #plt.grid(True)
    #plt.show()

    return ax
def plot_gnss_iteration(gnss_pos, gnss_ori, stixel_points):

    origin = ORIGIN
    line_strings = LINE_STRINGS

    plt.figure(figsize=(8, 8))

    for ls in line_strings:
        x, y = ls.xy
        x = [xi - origin[0] for xi in x]
        y = [yi - origin[1] for yi in y]
        plt.plot(x, y, color='black', linestyle='-')
    
    n = len(gnss_pos)

    gnss_x = [pt[0] for pt in gnss_pos]
    gnss_y = [pt[1] for pt in gnss_pos]

    alphas = [0.2 + 0.8*(i/(n-1)) if n > 1 else 1 for i in range(n)]

    for xi, yi, a in zip(gnss_x, gnss_y, alphas):
        plt.scatter(xi, yi, color=(1, 0, 0, a), s=25)


    r = R.from_quat(gnss_ori)
    euler_angles = r.as_euler('xyz', degrees=False)
    yaw = euler_angles[2]
    boat_position = gnss_pos[-1]    
    stixel_points_global = transform_stixel_points(stixel_points, boat_position, yaw)

    xs, ys = zip(*stixel_points_global)
    plt.fill(xs, ys, color='cyan', alpha=0.3, label="Free Space")
    plt.plot(xs, ys, marker='o', color='cyan')
    
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title("GNSS Data and Free Space")
    ax = plt.gca()
    ax.invert_yaxis()
    ax.invert_xaxis()
    #plt.grid(True)
    plt.show(block=False)
    plt.pause(1)  # Display the plot for a short period
    plt.close()

def plot_gnss_iteration_video(gnss_pos, gnss_ori, stixel_points):

    origin = ORIGIN
    line_strings = LINE_STRINGS

    #plt.figure(figsize=(8, 8))

    width, height = 1080, 1080
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(1,1,1)

    for ls in line_strings:
        x, y = ls.xy
        x = [xi - origin[0] for xi in x]
        y = [yi - origin[1] for yi in y]
        ax.plot(x, y, color='black', linestyle='-')
    
    n = len(gnss_pos)

    gnss_x = [pt[0] for pt in gnss_pos]
    gnss_y = [pt[1] for pt in gnss_pos]

    alphas = [0.2 + 0.8*(i/(n-1)) if n > 1 else 1 for i in range(n)]

    for xi, yi, a in zip(gnss_x, gnss_y, alphas):
        plt.scatter(xi, yi, color=(1, 0, 0, a), s=25)


    r = R.from_quat(gnss_ori)
    euler_angles = r.as_euler('xyz', degrees=False)
    yaw = euler_angles[2]
    boat_position = gnss_pos[-1]    
    stixel_points_global = transform_stixel_points(stixel_points, boat_position, yaw)

    xs, ys = zip(*stixel_points_global)
    ax.fill(xs, ys, color='cyan', alpha=0.3, label="Free Space")
    ax.plot(xs, ys, marker='o', color='cyan')
    
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("GNSS Data and Free Space")
    ax.invert_yaxis()
    ax.invert_xaxis()
    #plt.grid(True)
    #plt.show(block=False)
    #plt.pause(1)  # Display the plot for a short period
    #plt.close()

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.close(fig)
    return img_bgr



def transform_stixel_points(stixel_points, boat_position, heading):
    offset_y = TRANS_FLOOR_TO_LIDAR[0]
    offset_x = TRANS_FLOOR_TO_LIDAR[1]

    transformed_points = []

    heading_offset = 0 # math.radians(1.5)
    corrected_heading = - heading + math.pi + heading_offset
    cos_theta = math.cos(corrected_heading)
    sin_theta = math.sin(corrected_heading)
    for x, y in stixel_points:
        x_rel = x - offset_x
        y_rel = y - offset_y

        global_x = boat_position[0] + (x_rel * cos_theta - y_rel * sin_theta)
        global_y = boat_position[1] + (x_rel * sin_theta + y_rel * cos_theta)
        transformed_points.append((global_x, global_y))

    cam_pos_x = boat_position[0] - (offset_x * cos_theta - offset_y * sin_theta)
    cam_pos_y = boat_position[1] - (offset_x * sin_theta + offset_y * cos_theta)
    cam_pos = (cam_pos_x, cam_pos_y)

    transformed_points.append(cam_pos) # close the polygon
    transformed_points.append(transformed_points[0]) # close the polygon

    return transformed_points

def get_transform_piren_ned_from_vessel(self):
    ma2_tpose = self._get_ma2_tpose_ned()
    ma2_trans = ma2_tpose[1:4]
    ma2_rot_quat = ma2_tpose[4:]
    ma2_rot_mat = R.from_quat(ma2_rot_quat).as_matrix()

    H_points_piren_from_vessel = np.block([
        [ma2_rot_mat, ma2_trans[:,np.newaxis]], 
        [np.zeros((1,3)), np.ones((1,1))]
    ])
    return H_points_piren_from_vessel

def get_transform_enu_from_zed():
    H_points_piren_from_vessel = get_transform_piren_ned_from_vessel()
    H_POINTS_FLOOR_FROM_LEFT_CAM = invert_transformation(H_POINTS_LEFT_CAM_FROM_FLOOR)
    H_POINTS_LEFT_CAM_FROM_LEFT_ZED = invert_transformation(H_POINTS_LEFT_ZED_FROM_LEFT_CAM)
    H_points_enu_from_zed = H_POINTS_PIREN_ENU_FROM_PIREN @ H_points_piren_from_vessel @ H_POINTS_VESSEL_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LEFT_CAM @ H_POINTS_LEFT_CAM_FROM_LEFT_ZED
    return H_points_enu_from_zed

def get_zed_pose_enu():
    H_points_enu_from_zed = get_transform_enu_from_zed()

    H = H_points_enu_from_zed
    R_enu = H[0:3, 0:3]
    rot = R.from_matrix(R_enu)
    rot_quat = rot.as_quat()
    t_enu = H[0:3, 3]
    return t_enu, rot_quat

def invert_transformation(H):
    R = H[:3,:3]
    T = H[:3,3]
    H_transformed = np.block([ # ROT_FLOOR_TO_CAM.T, -ROT_FLOOR_TO_CAM.T.dot(TRANS_FLOOR_TO_CAM)[:,np.newaxis]
        [R.T, -R.T.dot(T)[:,np.newaxis]],
        [np.zeros((1,3)), np.ones((1,1))]
    ])
    return H_transformed




if __name__ == "__main__":
    file_path = "files/linestrings.json"
    line_strings = get_line_strings_from_file(file_path)
    plot_line_strings(line_strings, origin=[400000000, -50000])
    #plot_line_strings(line_strings)