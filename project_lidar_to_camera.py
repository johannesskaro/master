import cv2
import numpy as np
import matplotlib.pyplot as plt

#from o3d_pc_visualizer import O3DPointCloudVisualizer

def transform_lidar_to_camera_frame(R_IMU_TO_LIDAR, t_IMU_TO_LIDAR, R_IMU_TO_CAM, t_IMU_TO_CAM, K, points, intensity):

    rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8)/255, (points.shape[0], 1))

    intensity_clipped = np.clip(intensity, 0, 100)

    #plt.clf()
    #plt.hist(intensity_clipped, bins=100)
    #plt.pause(0.1)

    # Transform points from lidar to imu frame
    transform = np.block([
            [R_IMU_TO_LIDAR, t_IMU_TO_LIDAR[:,np.newaxis]], 
            [np.zeros((1,3)), np.ones((1,1))]
    ])
    points_imu = transform.dot(np.r_[points.T, np.ones((1, points.shape[0]))])[0:3, :].T

    # Transform points from floor to cam
    transform = np.block([
        [R_IMU_TO_CAM.T, -R_IMU_TO_CAM.T.dot(t_IMU_TO_CAM)[:,np.newaxis]], 
        [np.zeros((1,3)), np.ones((1,1))]
    ])
    points_c = transform.dot(np.r_[points_imu.T, np.ones((1, points_imu.shape[0]))])[0:3, :].T

    rvec = np.zeros((1,3), dtype=np.float32)
    tvec = np.zeros((1,3), dtype=np.float32)
    distCoeff = np.zeros((1,5), dtype=np.float32)
    image_points, _ = cv2.projectPoints(points_c, rvec, tvec, K, distCoeff)

    #image_points_forward = image_points[points_c[:,2] > 0]
    image_points_forward = image_points

    #image = image_points_to_image(image_points_forward, intensities=None)

    #o3d_vis.update(points_c, rgb)
    #cv2.imshow("Image", image)
    # print(f"{timestamp}")
    #cv2.waitKey(10)

    return image_points_forward, points_c


def merge_lidar_onto_image(image, lidar_points, lidar_3d_points=None, intensities=None, point_size=2, max_value=60, min_value=0):


    if intensities is not None and len(intensities.shape) == 2:
        intensities = np.squeeze(intensities, axis=1)  # From (N, 1) to (N,)

    image_with_lidar = image.copy()
    height, width = image.shape[:2]

    # Create a separate overlay for the lidar points
    lidar_overlay = np.zeros_like(image_with_lidar)

    if lidar_3d_points is not None:
        if lidar_3d_points.ndim == 1:
            depths = None  # Already 1D: each element is a depth value
        else:
            depths = lidar_3d_points[:, 2]
    else:
        depths = None

    # If intensities are provided, ensure they match the number of points
    if depths is not None:
        if len(depths) != len(lidar_points):
            raise ValueError("The length of intensities must match the number of lidar points.")

        # Normalize intensities
        if max_value is None:
            max_value = np.max(depths)
        if min_value is None:
            min_value = np.min(depths)

        # Avoid division by zero
        if max_value == min_value:
            max_value = min_value + 1

        depths_normalized = (depths - min_value) / (max_value - min_value)
        depths_normalized = np.clip(depths_normalized, 0, 1)
    else:
        # Use a default intensity of 1 for all points if no intensities are provided
        depths_normalized = np.ones(len(lidar_points))
    
    # Use the 'Reds' colormap
    #colormap = plt.get_cmap('Reds')
    colormap = plt.get_cmap('gist_earth')

    # Draw points on the lidar overlay image
    for i, point in enumerate(lidar_points):
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < width and 0 <= y < height:
            value_norm = depths_normalized[i]
            rgba = colormap(value_norm)  # returns RGBA, take RGB
            color = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
            cv2.circle(lidar_overlay, (x, y), point_size, color, -1)

            #color = tuple(int(c * 255) for c in color[::-1])  # convert to BGR
            #color = (0, 0, 255)
            #cv2.circle(lidar_overlay, (x, y), point_size, color, -1)

    # Blend the original image and the lidar overlay
    alpha = 1  # Weight of the original image
    beta = 0.8   # Weight of the overlay
    gamma = 0.0  # Scalar added to each sum
    image_with_lidar = cv2.addWeighted(image_with_lidar, alpha, lidar_overlay, beta, gamma)

    return image_with_lidar



def image_points_to_image(image_points, intensities = None):
    image_size = np.array((1080, 1920))
    scale = 5
    image_size_scaled = np.array(image_size/scale, dtype=np.uint)
    image = np.zeros(image_size_scaled, dtype=np.uint8)
    intensity_max = np.max(intensities)

    for i, point in enumerate(np.squeeze(image_points, axis=1)):
        x, y = np.array(point/scale, dtype=np.int32)
        if 0 <= x < image_size_scaled[1] and 0 <= y < image_size_scaled[0]:
            if intensities is not None:
                intensity = intensities[i] * 255 / intensity_max
                if image[y, x] == 0:
                    image[y, x] = intensity
                else:
                    image[y, x] = np.maximum(intensity, image[y, x])
            else:
                image[y, x] = 255
    
    image = cv2.resize(image, (image_size[1], image_size[0]))
    
    return image

