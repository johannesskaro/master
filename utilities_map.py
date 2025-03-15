import json
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.ndimage import rotate
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
def plot_gnss_iteration(gnss_pos, gnss_ori, stixel_points, pmo_list):

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



    #alphas = [0.3 + 0.4*(i/(n-1)) if n > 1 else 1 for i in range(n)]

    #for i, (xi, yi, a) in enumerate(zip(gnss_x, gnss_y, alphas)):
     #   if i == 0:
     #       plt.scatter(xi, yi, color='green', alpha=a, s=25, label="Ferry Position")
     #   else:
    #        plt.scatter(xi, yi, color='green', alpha=a, s=25)

    

    r = R.from_quat(gnss_ori)
    euler_angles = r.as_euler('xyz', degrees=False)
    yaw = euler_angles[2]
    boat_position = gnss_pos[-1]   

    plt.scatter(boat_position[0], boat_position[1], color='green', label="Ego Vessel") 


    boat_img = plt.imread("icons/ferry.png")
    rotation_angle = math.degrees(- yaw)
    rotated_boat_img = rotate(boat_img, angle=rotation_angle, reshape=True)
    rotated_boat_img = np.clip(rotated_boat_img, 0, 1)
    img_box = OffsetImage(rotated_boat_img, zoom=0.07)
    ab = AnnotationBbox(img_box, boat_position, frameon=False, box_alignment=(0.5, 0.5))
    ab.set_zorder(-10)

    stixel_points_global_poly = transform_stixel_points(stixel_points, boat_position, yaw)
    stixel_points_global = stixel_points_global_poly[:len(pmo_list)]

    xs, ys = zip(*stixel_points_global_poly)
    plt.fill(xs, ys, color='cyan', alpha=0.3, label="Free Space")
    #plt.plot(xs, ys, color='cyan')

    for (x, y), pmo in zip(stixel_points_global, pmo_list):
        if pmo == 1:
            plt.scatter(x, y, color='red', marker='o', s=25, label="Boats" if 'Boats' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(x, y, color='blue', marker='o', s=25, label="Static objects" if 'Static objects' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title("Free Space Estimation")
    ax = plt.gca()
    ax.add_artist(ab)
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.legend()
    plt.show(block=False)
    plt.pause(1)  # Display the plot for a short period
    plt.close()

def plot_gnss_iteration_video(gnss_pos, gnss_ori, stixel_points, pmo_list):

    origin = ORIGIN
    line_strings = LINE_STRINGS

    #fig = plt.figure(figsize=(8, 8))

    width, height = 1080, 1080
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(1,1,1)

    for ls in line_strings:
        x, y = ls.xy
        x = [xi - origin[0] for xi in x]
        y = [yi - origin[1] for yi in y]
        ax.plot(x, y, color='black', linestyle='-')
    
    #n = len(gnss_pos)

    #gnss_x = [pt[0] for pt in gnss_pos]
    #gnss_y = [pt[1] for pt in gnss_pos]

    #alphas = [0.2 + 0.8*(i/(n-1)) if n > 1 else 1 for i in range(n)]

    #for xi, yi, a in zip(gnss_x, gnss_y, alphas):
     #   plt.scatter(xi, yi, color=(1, 0, 0, a), s=25)


    r = R.from_quat(gnss_ori)
    euler_angles = r.as_euler('xyz', degrees=False)
    yaw = euler_angles[2]
    boat_position = gnss_pos[-1]

    boat_img = plt.imread("icons/ferry.png")
    rotation_angle = math.degrees(- yaw)
    rotated_boat_img = rotate(boat_img, angle=rotation_angle, reshape=True)
    rotated_boat_img = np.clip(rotated_boat_img, 0, 1)
    img_box = OffsetImage(rotated_boat_img, zoom=0.08)
    ab = AnnotationBbox(img_box, boat_position, frameon=False, box_alignment=(0.5, 0.5))
    ab.set_zorder(-10)



    stixel_points_global_poly = transform_stixel_points(stixel_points, boat_position, yaw)
    stixel_points_global = stixel_points_global_poly[:len(pmo_list)]

    xs, ys = zip(*stixel_points_global_poly)
    ax.fill(xs, ys, color='cyan', alpha=0.3, label="Free Space")
    ax.plot(xs, ys, color='cyan', zorder=-10)

    plt.scatter(boat_position[0], boat_position[1], s=50, color='green', label="Ego Vessel")

    for (x, y), pmo in zip(stixel_points_global, pmo_list):
        if pmo == 1:
            plt.scatter(x, y, color='red', marker='o', s=50, label="Boat" if 'Boat' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(x, y, color='blue', marker='o', s=50, label="Static Object" if 'Static Object' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    ax.set_xlabel("East [m]", fontsize=16)
    ax.set_ylabel("North [m]", fontsize=16)
    #ax.set_title("Free Space Estimation")
    ax.add_artist(ab)
    ax.invert_yaxis()
    ax.invert_xaxis()
    #plt.grid(True)
    #plt.show(block=False)
    #plt.pause(1)  # Display the plot for a short period
    #plt.close()

    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.close(fig)
    return img_bgr


def plot_previous_gnss_iterations(gnss_pos_list, gnss_ori_list, stixel_points_list):
    origin = ORIGIN
    line_strings = LINE_STRINGS

    plt.figure(figsize=(8, 8))

    for ls in line_strings:
        x, y = ls.xy
        x = [xi - origin[0] for xi in x]
        y = [yi - origin[1] for yi in y]
        #plt.plot(x, y, color='black', linestyle='-')
    
    n = len(gnss_pos_list[-1])
    gnss_x = [pt[0] for pt in gnss_pos_list]
    gnss_y = [pt[1] for pt in gnss_pos_list]

    alphas = [0.2 + 0.8*(i/(n-1)) if n > 1 else 1 for i in range(n)]

    for xi, yi, a in zip(gnss_x, gnss_y, alphas):
        plt.scatter(xi, yi, color=(0, 0, 1, a), s=50)

    
    cmap = plt.get_cmap("jet")
    num_scans = len(stixel_points_list)

    for i, stixel_points in enumerate(reversed(stixel_points_list)):

        pos = gnss_pos_list[-1-i]
        ori = gnss_ori_list[-1-i]
        r = R.from_quat(ori)
        euler_angles = r.as_euler('xyz', degrees=False)
        yaw = euler_angles[2]

        stixel_points_global = transform_stixel_points(stixel_points, pos, yaw)

        xs, ys = zip(*stixel_points_global)

        color = cmap(i/num_scans)

        scan_alpha = 0.1 + 0.5*(i/num_scans)
        #plt.fill(xs, ys, color='cyan', alpha=scan_alpha, label=f"Scan {num_scans-i}")
        plt.scatter(xs, ys, color='blue', s=50, alpha=scan_alpha, label=f"Scan {num_scans-i}")
        
        
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    #plt.title("GNSS Data and Free Space")
    ax = plt.gca()
    #ax.invert_yaxis()
    #ax.invert_xaxis()
    #plt.grid(True)
    plt.legend()
    plt.show(block=False)
    plt.pause(1)  # Display the plot for a short period
    plt.close()


def plot_previous_gnss_iterations_local(gnss_pos_list, gnss_ori_list, stixel_points_list):

    offset_y = - TRANS_FLOOR_TO_LIDAR[0]
    offset_x = - TRANS_FLOOR_TO_LIDAR[1]

    plt.figure(figsize=(8, 8))

    curr_boat_pos = gnss_pos_list[-1]
    curr_boat_ori = gnss_ori_list[-1]
    r = R.from_quat(curr_boat_ori)
    euler_angles = r.as_euler('xyz', degrees=False)
    curr_heading = euler_angles[2]

    n = len(gnss_pos_list[-1])
    cam_pos_list = []

    for i, gnss_pos in enumerate(gnss_pos_list):

        ori = gnss_ori_list[i]
        r = R.from_quat(ori)
        euler_angles = r.as_euler('xyz', degrees=False)
        heading = euler_angles[2]
        
        corrected_heading = - heading + math.pi

        R_corrected = rotation_matrix(corrected_heading)

        offset_vec = np.array([offset_x, offset_y])
        rotated_offset = R_corrected.dot(offset_vec)

        curr_cam_pos_x = curr_boat_pos[0] + rotated_offset[0]
        curr_cam_pos_y = curr_boat_pos[1] + rotated_offset[1]

        cam_pos_x_global = gnss_pos[0] + rotated_offset[0]
        cam_pos_y_global = gnss_pos[1] + rotated_offset[1]

        rel_x = cam_pos_x_global - curr_cam_pos_x
        rel_y = cam_pos_y_global - curr_cam_pos_y

        rel_vec = np.array([rel_x, rel_y])
        R_heading = rotation_matrix(curr_heading)
        rotated_rel = R_heading.dot(rel_vec)
        cam_x_local, cam_y_local = - rotated_rel

        cam_pos_list.append((cam_x_local, cam_y_local))


    gnss_x = [pt[0] for pt in cam_pos_list]
    gnss_y = [pt[1] for pt in cam_pos_list]
    
    for xi, yi in zip(gnss_x, gnss_y):
        plt.scatter(xi, yi, color='green', s=10)

    
    cmap = plt.get_cmap("jet")
    num_scans = len(stixel_points_list)

    curr_pos = gnss_pos_list[-1]
    curr_ori = gnss_ori_list[-1]
    r = R.from_quat(curr_ori)
    euler_angles = r.as_euler('xyz', degrees=False)
    curr_yaw = euler_angles[2]

    for i, stixel_points in enumerate(stixel_points_list):

        pos = gnss_pos_list[i]
        ori = gnss_ori_list[i]
        r = R.from_quat(ori)
        euler_angles = r.as_euler('xyz', degrees=False)
        yaw = euler_angles[2]

        stixel_points_local = transform_stixel_points_local(stixel_points, pos, yaw, curr_pos, curr_yaw)
        #stixel_points_local = stixel_points

        xs, ys = zip(*stixel_points_local)
        color = cmap(i/num_scans)

        scan_alpha = 0.01 + 0.3*(i/num_scans)**3
        #plt.fill(xs, ys, color='cyan', alpha=scan_alpha, label=f"Scan {num_scans-i}")
        #plt.scatter(xs, ys, color='blue', s=50, alpha=scan_alpha, label=f"Scan {num_scans-i}")
        label = "Stixels" if i == len(stixel_points_list) - 1 else ""
        plt.scatter(xs, ys, color='blue', s=10, alpha=scan_alpha, label=label)


    plt.scatter(0, 0, color='green', marker='*', s=200, label='Camera position')

        
    plt.xlabel("Z (m)")
    plt.ylabel("X (m)")
    #plt.title("GNSS Data and Free Space")
    ax = plt.gca()

    ax.set_xlabel("X [m]", fontsize=16)
    ax.set_ylabel("Z [m]", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)



    plt.grid(True)
    plt.savefig("images/bev_consistency_200_frames_v7.png", dpi=300, bbox_inches='tight')


    #plt.ioff()
    #plt.show()
    plt.close()
    #plt.show(block=False)
    #plt.pause(1)  # Display the plot for a short period
    #plt.close()




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

def transform_stixel_points_local(stixel_points, prev_pos, prev_heading, curr_pos, curr_heading):
    offset_y = - TRANS_FLOOR_TO_LIDAR[0]
    offset_x = - TRANS_FLOOR_TO_LIDAR[1]
    cam_offset = np.array([offset_x, offset_y])

    transformed_points = []

    R_prev = rotation_matrix(prev_heading)
    R_curr = rotation_matrix(curr_heading)


    prev_cam_global = np.array(prev_pos) + R_prev.dot(np.array(cam_offset))
    curr_cam_global = np.array(curr_pos) + R_curr.dot(np.array(cam_offset))

    for x, y in stixel_points:

        p_prev_cam = np.array([x, y])

        p_global = prev_cam_global + R_prev.dot(p_prev_cam)

        p_current_cam = R_curr.T.dot(p_global - curr_cam_global)
        transformed_points.append(tuple(p_current_cam))


    return transformed_points

def rotation_matrix(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta),  math.cos(theta)]])

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