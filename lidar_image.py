from shapely.geometry import Polygon, Point
import numpy as np
import cv2



def create_lidar_depth_image(xyz_c, K):

    IMAGE_WIDTH = 2048
    IMAGE_HEIGHT = 64
    IMAGE_HEIGHT_CAMERA = 1080

    POINTS_PER_SCANLINE = 2048
    NUM_SCANLINES = 64

    xyz_c_reshaped = xyz_c.reshape(NUM_SCANLINES, POINTS_PER_SCANLINE, 3)

    rvec = np.zeros((1,3), dtype=np.float32)
    tvec = np.zeros((1,3), dtype=np.float32)
    distCoeff = np.zeros((1,5), dtype=np.float32)
    image_polygon = Polygon([(0, 0), (IMAGE_WIDTH, 0), (IMAGE_WIDTH, IMAGE_HEIGHT_CAMERA), (0, IMAGE_HEIGHT_CAMERA)])

    lidar_depth_image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), np.nan, dtype=np.float32)

    scanline_to_img_row = np.full(IMAGE_HEIGHT, -1, dtype=int)
    img_row_to_scanline = np.full(IMAGE_HEIGHT_CAMERA, -1, dtype=int)

    for row_idx in range(IMAGE_HEIGHT):
        row_points = xyz_c_reshaped[row_idx]
        image_points, _ = cv2.projectPoints(row_points, rvec, tvec, K, distCoeff)
        image_points = image_points.squeeze()

        mask_forward = row_points[:,2] > 0
        row_points_forward = row_points[mask_forward]
        image_points_forward = image_points[mask_forward]
        
        inside_indices = np.array([i for i, pt in enumerate(image_points_forward) if image_polygon.contains(Point(pt))], dtype=int)

        row_filtered_image_points = image_points_forward[inside_indices]
        row_filtered_xyz_c = row_points_forward[inside_indices]

        if row_filtered_image_points.shape[0] == 0:
            continue  # Skip if no valid points in this row

        scanline_to_img_row[row_idx] = int(np.nanmedian(row_filtered_image_points[:,1]))

        # Normalize x-coordinates to range [0, 1920]
        col_indices = np.clip((row_filtered_image_points[:, 0] / IMAGE_WIDTH) * IMAGE_WIDTH, 0, IMAGE_WIDTH - 1).astype(int)

        for i, col in enumerate(col_indices):
            depth = row_filtered_xyz_c[i, 2]  # Use Z (depth)
            if np.isnan(lidar_depth_image[row_idx, col]) or depth < lidar_depth_image[row_idx, col]:
                lidar_depth_image[row_idx, col] = depth  # Keep closest point
        
    valid_mask = scanline_to_img_row >= 0
    if not np.any(valid_mask):
        raise ValueError("No valid scanlines available for mapping!")
    valid_scanline_indices = np.where(valid_mask)[0]
    valid_scanline_img_rows = scanline_to_img_row[valid_mask]

    all_img_rows = np.arange(IMAGE_HEIGHT_CAMERA)
    differences = np.abs(all_img_rows[:, None] - valid_scanline_img_rows[None, :])
    nearest_valid_idx = np.argmin(differences, axis=1)
    img_row_to_scanline = valid_scanline_indices[nearest_valid_idx]


    return lidar_depth_image, scanline_to_img_row, img_row_to_scanline