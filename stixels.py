import numpy as np
from shapely.geometry import Polygon
from collections import deque
import utilities as ut
import cv2
from plotting import *
from scipy.interpolate import griddata, Rbf
from scipy import stats
from fastSAM import FastSAMSeg

class Stixels:

    N = 10
    stixel_2d_points_N_frames = deque(maxlen=N)
    
    fused_stixel_depth_list = []

    def __init__(self, num_of_stixels = 192) -> None:  #96 stixels
        self.num_stixels = num_of_stixels # 958//47 #gir 20I wan 
        self.rectangular_stixel_list = [[0, 0, 0, 0] for _ in range(num_of_stixels)]
        self.prev_rectangular_stixel_list = self.rectangular_stixel_list

    def get_stixel_2d_points_N_frames(self) -> np.array:
        return self.stixel_2d_points_N_frames

    def add_stixel_2d_points(self, stixel_2d_points: np.array) -> None:
        self.stixel_2d_points_N_frames.append(stixel_2d_points)

    def get_stixel_width(self, img_width) -> int:
        self.stixel_width = int(img_width // self.num_stixels)
        return self.stixel_width
    
    def get_free_space_boundary(self, water_mask: np.array) -> np.array:
        height, width = water_mask.shape
        stixel_width = self.get_stixel_width(width)

        search_height = height - 50
        submask = water_mask[:search_height, :]  
        reversed_mask = submask[::-1, :]  

        cond = (reversed_mask == 0)
        
        first_free_idx = np.argmax(cond, axis=0)  # shape: (width,)
        found = np.any(cond, axis=0)              # shape: (width,), True if free pixel found

        free_space_boundary = np.full(width, height, dtype=int)
        free_space_boundary[found] = search_height - 1 - first_free_idx[found]

        free_space_boundary_mask = np.zeros_like(water_mask, dtype=np.uint8)
        cols = np.arange(width)
        valid = free_space_boundary < height  # Only update columns where a valid boundary was found.
        free_space_boundary_mask[free_space_boundary[valid], cols[valid]] = 1

        free_space_boundary[0] = free_space_boundary[1] # set first stixel as the same as the second

        return free_space_boundary, free_space_boundary_mask

    def get_stixels_base(self, water_mask: np.array) -> np.array:

        stixel_positions = np.zeros((self.num_stixels, 2))

        free_space_boundary, free_space_boundary_mask = self.get_free_space_boundary(water_mask)
        height, width = water_mask.shape
        stixel_width = int(width // self.num_stixels)
        stixel_mask = np.zeros_like(water_mask)

        for n in range(self.num_stixels):

            stixel = free_space_boundary[n * stixel_width:(n + 1) * stixel_width]
            stixel_y_pos = int(np.mean(stixel))
            stixel_x_pos = n * stixel_width + stixel_width // 2
            stixel_pos = np.array([stixel_x_pos, stixel_y_pos])
            #stixel_positions = np.vstack([stixel_positions, stixel_pos])
            stixel_positions[n] = stixel_pos
            stixel_mask[stixel_y_pos, n * stixel_width:(n + 1) * stixel_width] = 1
            #stixel_mask[stixel_y_pos, stixel_x_pos] = 1

        #print(stixel_positions)
        return stixel_mask, stixel_positions
    
    def create_rectangular_stixels(self, water_mask, disparity_map, depth_map):
        free_space_boundary, _ = self.get_free_space_boundary(water_mask)
        stixel_width = self.get_stixel_width(water_mask.shape[1])

        std_dev_threshold = 0.35
        #median_disp_change_threshold = 0.1
        window_size = 10
        min_stixel_height = 20

        rectangular_stixel_mask = np.zeros_like(water_mask)
        self.rectangular_stixel_list = []

        for n in range(self.num_stixels):

            stixel_range = slice(n * stixel_width, (n + 1) * stixel_width)
            stixel_base = free_space_boundary[stixel_range]
            stixel_base_height = int(np.median(stixel_base))
            stixel_top_height = stixel_base_height - min_stixel_height

            std_dev = 0
            median_row_disp_list = []

            for v in range(stixel_base_height, 0, -1):

                median_row_disp = np.nanmedian(disparity_map[v, stixel_range])
                median_row_disp_list.append(median_row_disp)
                std_dev = np.std(median_row_disp_list)
                #print(f"std_dev: {std_dev}")
                if std_dev > std_dev_threshold:
                    stixel_top_height = v
                    if stixel_base_height - v < min_stixel_height:
                        stixel_top_height = stixel_base_height - min_stixel_height
                    break

            stixel_median_disp = np.nanmedian(disparity_map[stixel_top_height:stixel_base_height, stixel_range])
            stixel_median_depth = np.nanmedian(depth_map[stixel_top_height:stixel_base_height, stixel_range])
            stixel = [stixel_top_height, stixel_base_height, stixel_median_disp, stixel_median_depth]
            self.rectangular_stixel_list.append(stixel)
            rectangular_stixel_mask[stixel_top_height:stixel_base_height, stixel_range] = 1

        return self.rectangular_stixel_list, rectangular_stixel_mask
    
    
    def create_rectangular_stixels_2(self, water_mask, disparity_map, depth_map):
        free_space_boundary, _ = self.get_free_space_boundary(water_mask)
        stixel_width = self.get_stixel_width(water_mask.shape[1])

        std_dev_threshold_base = 0.25   # 0.35
        ref_depth = 40
        offset = 0.2
        min_stixel_height = 20

        rectangular_stixel_mask = np.zeros_like(water_mask)
        self.rectangular_stixel_list = []

        for n in range(self.num_stixels):
            stixel_range = slice(n * stixel_width, (n + 1) * stixel_width)
            stixel_base = free_space_boundary[stixel_range]
            v_f = int(np.median(stixel_base))
            v_top = v_f - min_stixel_height

            # Forhåndsberegn rad-medianer for de aktuelle kolonnene (hele bildet)
            stixel_disparity = disparity_map[:, stixel_range]
            row_medians = np.nanmedian(stixel_disparity, axis=1)

            stixel_depth = depth_map[v_f, stixel_range]
            base_depth = np.nanmedian(stixel_depth)

            adaptive_threshold = std_dev_threshold_base * (base_depth / ref_depth) + offset

            mean = 0.0
            M2 = 0.0
            count = 0
            std_dev_cost = 0.0

            for v in range(v_f, -1, -1):
                x = row_medians[v]
                count += 1
                delta = x - mean
                mean += delta / count
                delta2 = x - mean
                M2 += delta * delta2

                if count > 1:
                    current_std = np.sqrt(M2 / (count - 1))
                    
                    
                    #if current_std > adaptive_threshold:
                    #    v_top = v
                    #    if (v_f - v) < min_stixel_height:
                    #        v_top = v_f - min_stixel_height
                    #    break

                    std_dev_cost = 2**(1 - 2*(current_std**2))

                    if std_dev_cost < 1.5:
                        v_top = v
                        if (v_f - v) < min_stixel_height:
                            v_top = v_f - min_stixel_height
                        break

            stixel_median_disp = np.nanmedian(disparity_map[v_top:v_f, stixel_range])
            stixel_median_depth = np.nanmedian(depth_map[v_top:v_f, stixel_range])

            stixel = [v_top, v_f, stixel_median_disp, stixel_median_depth]
            self.rectangular_stixel_list.append(stixel)

            rectangular_stixel_mask[v_top:v_f, stixel_range] = 1

        return self.rectangular_stixel_list, rectangular_stixel_mask
    
    def create_rectangular_stixels_3(self, water_mask, disparity_map, depth_map, sam_countours):

        self.prev_rectangular_stixel_list = self.rectangular_stixel_list
        free_space_boundary, _ = self.get_free_space_boundary(water_mask)
        stixel_width = self.get_stixel_width(water_mask.shape[1])
        height, width = disparity_map.shape
        min_stixel_height = 20
        max_stixel_height = 300

        cost_map, free_space_boundary_depth = self.create_cost_map_2(disparity_map, depth_map, free_space_boundary, sam_countours)

        top_boundary, boundary_mask = self.get_optimal_height(cost_map, free_space_boundary_depth, free_space_boundary)

        #boundary_mask = cv2.resize(boundary_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("Boundary Mask", boundary_mask.astype(np.uint8) * 255)


        #top_boundary, boundary_mask = self.get_greedy_height(cost_map)
        
        rectangular_stixel_mask = np.zeros_like(water_mask)
        self.rectangular_stixel_list = []

        for n in range(self.num_stixels):

            stixel_range = slice(n * stixel_width, (n + 1) * stixel_width)
            v_top = top_boundary[n]
            stixel_base = free_space_boundary[stixel_range]
            v_f = int(np.median(stixel_base))

            if (v_f - v_top) < min_stixel_height:
                v_top = v_f - min_stixel_height
            elif (v_f - v_top) > max_stixel_height:
                v_top = v_f - max_stixel_height

            

            stixel_median_depth = np.nanmedian(depth_map[v_top:v_f, stixel_range])
            stixel_median_disp = np.nanmedian(disparity_map[v_top:v_f, stixel_range])
            stixel = [v_top, v_f, stixel_median_disp, stixel_median_depth]
            self.rectangular_stixel_list.append(stixel)

        self.rectangular_stixel_list[0] = self.rectangular_stixel_list[1]

        return self.rectangular_stixel_list, rectangular_stixel_mask
    

    
    def create_cost_map_2(self, disparity_map, depth_map, free_space_boundary, sam_contours):
        height, width = disparity_map.shape

        # Preprocessing steps
        normalized_disparity = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
        normalized_disparity = normalized_disparity.astype(np.uint8)
        blurred_image = cv2.GaussianBlur(normalized_disparity, (5, 5), 0)
        grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        grad_y = cv2.convertScaleAbs(grad_y)
        _, grad_y = cv2.threshold(grad_y, 200, 255, cv2.THRESH_BINARY)
        grad_y = grad_y.astype(np.float32) / 255.0

        grad_y = ut.filter_mask_by_boundary(grad_y, free_space_boundary, offset=10)
        #cv2.imshow("grad_y", grad_y.astype(np.uint8) * 255)
        grad_y = ut.get_bottommost_line(grad_y)

        sam_contours = ut.filter_mask_by_boundary(sam_contours, free_space_boundary, offset=10)
        #cv2.imshow("sam_contours", sam_contours.astype(np.uint8))
        sam_contours = sam_contours.astype(np.float32) / 255.0
        sam_contours = ut.get_bottommost_line(sam_contours)

        cost_map = np.full((height, self.num_stixels), 255, dtype=float)
        free_space_boundary_depth = np.zeros((height, self.num_stixels))
        prev_stixels = self.rectangular_stixel_list

        for n in range(self.num_stixels):
            stixel_range = slice(n * self.stixel_width, (n + 1) * self.stixel_width)
            stixel_base = free_space_boundary[stixel_range]
            v_f = int(np.median(stixel_base))

            stixel_disparity = disparity_map[:, stixel_range]
            row_medians = np.nanmedian(stixel_disparity, axis=1)

            row_medians_rev = row_medians[:v_f+1][::-1]
            
            # Precompute cumulative sums for row_medians for vectorized std dev over a sliding window
            cumsum = np.cumsum(row_medians_rev)
            cumsum2 = np.cumsum(row_medians_rev**2)
            
            # Precompute grad_y means for stixel_range
            grad_y_means = np.mean(grad_y[:, stixel_range], axis=1)
            #grad_y_means = np.where(np.mean(grad_y[:, stixel_range], axis=1) > 0.3, 1, 0)
            #grad_y_means = np.where(np.any(grad_y[:, stixel_range] > 0, axis=1), 1, 0)
            #sam_contours_means = np.mean(sam_contours[:, stixel_range], axis=1)
            sam_contours_means = np.where(np.mean(sam_contours[:, stixel_range], axis=1) > 0.5, 1, 0)
            #sam_contours_means = np.where(np.any(sam_contours[:, stixel_range] > 0, axis=1), 1, 0)
            
            depth_window = depth_map[v_f-10:v_f+1, stixel_range]
            free_space_boundary_depth[v_f, n] = np.nanmedian(depth_window)


            grad_y_means[v_f-10:v_f+1] = 0
            sam_contours_means[v_f-30:v_f+1] = 0
            prev_v_top = prev_stixels[n][0]
            prev_depth = prev_stixels[n][3]

            # Define parameters (tuning these is key)
            w1, w2, w3, w4, w5, w6 = 200, 100, 0, 0, 0, 0 #100, 100, 0, 150, 0, 0
            local_window = 5
            d_hat = row_medians[v_f]
            Delta_D = 1

            # Instead of a Python loop for each row, consider vectorizing the range from 0 to v_f
            # For each v in [0, v_f], compute the local window statistics
            for v in range(v_f, -1, -1):
                i = v_f - v

                # Compute cumulative mean and std via the cumulative arrays:
                count = i + 1
                mean = cumsum[i] / count
                var = (cumsum2[i] / count) - (mean**2)
                current_std = np.sqrt(var) if var > 0 else 0
                
                #local_std = self.compute_local_std(cumsum, cumsum2, i, local_window)
                local_std = 0

                high_local_std = 1 if local_std > 0.15 and count > 10 else 0
                high_std = 1 if current_std > 0.35 else 0

                # Height cost based on difference from previous stixel top
                if prev_v_top == 0:
                    delta_height = 0
                    height_cost = 0
                else:
                    delta_height = abs(v - prev_v_top)
                    #scale = max(0, 1 - (abs(prev_depth - free_space_boundary_depth[v_f, n]) / 5))
                    height_cost = delta_height #* scale

                grad_y_v = grad_y_means[v]
                std_dev_cost = 2**(1 - 2*(current_std**2))

                sam_contour = sam_contours_means[v]
        
                
                cost_map[v, n] = - w1 * grad_y_v - w2 * std_dev_cost + w3 * high_std - w4 * high_local_std + w5 * height_cost - w6 * sam_contour
                #cost_map[v, n] = - w1 * grad_y_v

        cost_map = cv2.normalize(cost_map, None, 0, 255, cv2.NORM_MINMAX)
        cost_map = cost_map.astype(np.uint8)

        cost_map_resized = cv2.resize(-cost_map, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Cost Map", cost_map_resized)
        #cv2.imshow("grad_y", grad_y.astype(np.uint8) * 255)
        #colored_disparity = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET)
        #cv2.imshow("colored_disparity", colored_disparity)

        return cost_map, free_space_boundary_depth
    
    def compute_local_std(self, cumsum, cumsum2, i, local_window):
        # Define the window start index:
        window_start = max(0, i - local_window + 1)
        L = i - window_start + 1

        # Sum over the window:
        window_sum = cumsum[i] - (cumsum[window_start - 1] if window_start > 0 else 0)
        window_sum2 = cumsum2[i] - (cumsum2[window_start - 1] if window_start > 0 else 0)
        
        local_mean = window_sum / L
        local_variance = (window_sum2 / L) - (local_mean ** 2)
        return np.sqrt(local_variance) if local_variance > 0 else 0
                
    
    def interpolate_depth_image(self, lidar_depth_image, method='linear'):
        height, width = lidar_depth_image.shape
        # Create coordinate grids.
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Flatten the grids and the image.
        points = np.vstack((grid_y.ravel(), grid_x.ravel())).T
        values = lidar_depth_image.ravel()
        
        # Identify valid points (non-NaN).
        valid_mask = ~np.isnan(values)
        valid_points = points[valid_mask]
        valid_values = values[valid_mask]

        if method == 'rbf':
            # RBF expects separate coordinate arrays.
            y_valid = valid_points[:, 0]
            x_valid = valid_points[:, 1]
            rbf = Rbf(x_valid, y_valid, valid_values, function='multiquadric', epsilon=2, smooth=0)
            interpolated = rbf(grid_x, grid_y)

        elif method == 'linear':

            interpolated = griddata(valid_points, valid_values, (grid_y, grid_x), method='linear')
            # Optionally, fill remaining NaNs with nearest neighbor interpolation:
            # nan_mask = np.isnan(interpolated)
            # interpolated[nan_mask] = griddata(valid_points, valid_values, (grid_y[nan_mask], grid_x[nan_mask]), method='nearest')

        elif method == 'nearest':
            
            interpolated = griddata(valid_points, valid_values, (grid_y, grid_x), method='nearest')

        elif method == 'cubic':
            interpolated = griddata(valid_points, valid_values, (grid_y, grid_x), method='cubic')

        else:
            raise ValueError("Unknown interpolation method: choose 'rbf' or 'linear'")
        
        # Optionally, fill remaining NaNs with a nearest-neighbor interpolation.
        nan_mask = np.isnan(interpolated)
        interpolated[nan_mask] = griddata(valid_points, valid_values, (grid_y[nan_mask], grid_x[nan_mask]), method='nearest')
    
        return interpolated
    
    def create_stixels_from_lidar_depth_image(self, lidar_depth_image, scanline_to_img_row, img_row_to_scanline, free_space_boundary):

        height, width = lidar_depth_image.shape
        stixel_width = self.get_stixel_width(width)
        std_dev_threshold = 0.35
        self.rectangular_stixel_list = []

  
        filled_lidar_depth_image = self.interpolate_depth_image(lidar_depth_image, method='nearest')


        for n in range(self.num_stixels):
            stixel_range = slice(n * stixel_width, (n + 1) * stixel_width)
            stixel_base = free_space_boundary[stixel_range]
            stixel_base_height = int(np.median(stixel_base))
            v_f = img_row_to_scanline[stixel_base_height]
            v_top = v_f

            stixel_depth = lidar_depth_image[:, stixel_range]
            row_medians = np.nanmedian(stixel_depth, axis=1)

            indices = np.arange(len(row_medians))
            valid_mask = ~np.isnan(row_medians)

            if np.any(valid_mask):
                # Interpolate to fill in the NaN values.
                row_medians_filled = np.interp(indices, indices[valid_mask], row_medians[valid_mask])
            else:
                # Fallback: if no valid data exists, you might choose to keep the array as is or set a default.
                row_medians_filled = row_medians

            mean = 0.0
            M2 = 0.0
            count = 0

            for v in range(v_f - 1, -1, -1):
                x = row_medians_filled[v]

                if np.isnan(x):
                    continue  # Skip rows without valid data

                count += 1

                delta = x - mean
                mean += delta / count
                delta2 = x - mean
                M2 += delta * delta2

                if count > 1:
                    current_std = np.sqrt(M2 / (count - 1))
                    if current_std > std_dev_threshold:
                        v_top = v + 1
                        break

            stixel_median_depth = np.nanmedian(lidar_depth_image[v_top:v_f, stixel_range])
            stixel = [scanline_to_img_row[v_top], stixel_base_height, -1, stixel_median_depth]
            self.rectangular_stixel_list.append(stixel)


        # Visualize the stixels

            lidar_depth_image[:, (n) * stixel_width] = 100
            filled_lidar_depth_image[:, (n) * stixel_width] = 100

        for u in range(width):
            img_v_f = int(free_space_boundary[u])
            v_f = img_row_to_scanline[img_v_f]
            lidar_depth_image[v_f, u] = 100
            filled_lidar_depth_image[v_f, u] = 100
            filled_lidar_depth_image[v_f:height, u] = 0

        show_lidar_image(lidar_depth_image, "LiDAR Depth Image")
        show_lidar_image(filled_lidar_depth_image, "Filled LiDAR Depth Image")

        return self.rectangular_stixel_list

    def create_stixels_from_lidar_depth_image_2(self, lidar_depth_image, scanline_to_img_row, img_row_to_scanline, free_space_boundary):

        height, width = lidar_depth_image.shape
        stixel_width = self.get_stixel_width(width)
        scaling = 1

        #lidar_depth_image = self.interpolate_depth_image(lidar_depth_image, method='nearest')

        membership_image = np.zeros((height, width))

        for u in range(width):
            img_v_f = int(free_space_boundary[u])
            v_f = img_row_to_scanline[img_v_f]
            z_hat = lidar_depth_image[v_f, u]

            for v in reversed(range(v_f)):
                if np.isnan(lidar_depth_image[v, u]):
                    continue
                if np.isnan(z_hat):
                    z_hat = lidar_depth_image[v, u]
                    continue

                z_uv = lidar_depth_image[v, u]
                exponent = 1 - ((z_uv - z_hat) / scaling)**2
                membership_image[v, u] = 2**exponent - 1

        cost_image = np.zeros((height, width))

        top_boundary = np.zeros(width, dtype=int)
        boundary_mask_greedy = np.zeros((height, width), dtype=int)   
        
        for u in range(width):
            img_v_f = int(free_space_boundary[u])
            v_f = img_row_to_scanline[img_v_f]

            prefix = np.zeros(v_f + 2, dtype=np.float32)

            for v in range(v_f + 1):
                prefix[v + 1] = prefix[v] + membership_image[v, u]

            for v in range(v_f + 1):
                sum_above = prefix[v]
                sum_below = prefix[v_f + 1] - prefix[v]
                cost_image[v, u] = (sum_above - sum_below)

            best_row = np.argmin(cost_image[:, u])
            if best_row == 0:
                best_row = v_f
            top_boundary[u] = best_row
            boundary_mask_greedy[best_row, u] = 1

            lidar_depth_image[v_f+1:height, u] = 0

        top_boundary, boundary_mask = self.get_optimal_height(cost_image, lidar_depth_image, free_space_boundary, img_row_to_scanline)
        #top_boundary, boundary_mask = self.get_greedy_height(cost_image)

        self.rectangular_stixel_list = []

        for n in range(self.num_stixels):

            stixel_range = slice(n * stixel_width, (n + 1) * stixel_width)
            stixel_top = top_boundary[stixel_range]
            v_top = int(np.median(stixel_top))
            stixel_base = free_space_boundary[stixel_range]
            stixel_base_height = int(np.median(stixel_base))
            v_f = img_row_to_scanline[stixel_base_height]

            stixel_median_depth = np.nanmedian(lidar_depth_image[v_top:v_f, stixel_range])
            stixel = [scanline_to_img_row[v_top], stixel_base_height, -1, stixel_median_depth]
            self.rectangular_stixel_list.append(stixel)

        #       Visualize the stixels
        for u in range(width):
            img_v_f = int(free_space_boundary[u])
            v_f = img_row_to_scanline[img_v_f]
            lidar_depth_image[v_f:height, u] = 0
            lidar_depth_image[v_f, u] = 100
            

        show_lidar_image(cost_image, "Cost Image")
        show_lidar_image(membership_image, "Membership Image")
        show_lidar_image(lidar_depth_image, "LiDAR Depth Image")

        return self.rectangular_stixel_list
    
    def create_stixels_from_lidar_depth_image_3(self, lidar_depth_image, scanline_to_img_row, img_row_to_scanline, free_space_boundary):
        
        lidar_depth_image = self.interpolate_depth_image(lidar_depth_image, method="nearest")
        show_lidar_image(lidar_depth_image)

        lidar_depth_image = np.array(lidar_depth_image, dtype=np.float32)

        # Normalize depth values to range [0, 255]
        depth_min = np.nanmin(lidar_depth_image)  # Avoid NaN issues
        depth_max = np.nanmax(lidar_depth_image)

        if depth_max > depth_min:  # Prevent division by zero
            normalized_depth = (lidar_depth_image - depth_min) / (depth_max - depth_min) * 255
        else:
            normalized_depth = np.zeros_like(lidar_depth_image)

        lidar_depth_image_8bit = normalized_depth.astype(np.uint8)

        #edge_detection = cv2.Canny(lidar_depth_image_8bit, 100, 200)

        blurred_image = cv2.GaussianBlur(lidar_depth_image_8bit, (5, 5), 0)
        grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        edge_detection = cv2.convertScaleAbs(grad_y)
        _, edge_detection = cv2.threshold(edge_detection, 100, 255, cv2.THRESH_BINARY)

        height, width = lidar_depth_image.shape

        stixel_width = self.get_stixel_width(width)
        self.rectangular_stixel_list = []

        for n in range(self.num_stixels):
            stixel_range = slice(n * stixel_width, (n + 1) * stixel_width)
            stixel_base = free_space_boundary[stixel_range]
            stixel_base_height = int(np.median(stixel_base))
            v_f = img_row_to_scanline[stixel_base_height]
            v_top = v_f

            for v in range(v_f - 1, -1, -1):

                if np.median(edge_detection[v, stixel_range]) > 0:
                    v_top = v
                    break

            stixel_median_depth = np.nanmedian(lidar_depth_image[v_top:v_f, stixel_range])
            stixel = [scanline_to_img_row[v_top], stixel_base_height, -1, stixel_median_depth]
            self.rectangular_stixel_list.append(stixel)


        show_lidar_image(edge_detection, "Edge Detection")

        return self.rectangular_stixel_list


    def create_stixels(self, disparity_map, depth_map, free_space_boundary, cam_params):
        height, width = disparity_map.shape
        b = cam_params["b"]
        fx = cam_params["fx"]
        Delta_Z = 1 #2
        min_stixel_height = 20
        membership_image = np.zeros((height, width))
   
        self.rectangular_stixel_list = []

        for u in range(width):
            v_f = int(free_space_boundary[u])
            d_hat = disparity_map[v_f, u]
            z_u = depth_map[v_f, u]
            Delta_D = d_hat - (fx * b) / (z_u + Delta_Z)

            for v in range(height):
                d_uv = disparity_map[v, u]
                exponent = 1 - ((d_uv - d_hat) / Delta_D)**2
                membership_image[v, u] = 2**exponent - 1

        normalized = (membership_image - membership_image.min()) / (membership_image.max() - membership_image.min())

        cost_image = np.zeros((height, width))

        top_boundary = np.zeros(width, dtype=int)
        boundary_mask_greedy = np.zeros((height, width), dtype=int)   
        
        for u in range(width):
            v_f = int(free_space_boundary[u])
            prefix = np.zeros(v_f + 2, dtype=np.float32)

            for v in range(v_f + 1):
                prefix[v + 1] = prefix[v] + membership_image[v, u]

            for v in range(v_f + 1):
                sum_above = prefix[v]
                sum_below = prefix[v_f + 1] - prefix[v]
                cost_image[v, u] = (sum_above - sum_below)

            best_row = np.argmin(cost_image[:, u])
            if best_row == 0:
                best_row = v_f
            top_boundary[u] = best_row
            boundary_mask_greedy[best_row, u] = 1

        top_boundary, boundary_mask = self.get_optimal_height(cost_image, depth_map, free_space_boundary)

        cost_image = -cost_image
        normalized_cost = (cost_image - cost_image.min()) / (cost_image.max() - cost_image.min())

        stixel_width = self.get_stixel_width(width)
        rectangular_stixel_mask = np.zeros_like(disparity_map)

        for n in range(self.num_stixels):

            stixel_range = slice(n * stixel_width, (n + 1) * stixel_width)
            stixel_top = top_boundary[stixel_range]
            stixel_top_height = int(np.median(stixel_top))
            stixel_base = free_space_boundary[stixel_range]
            stixel_base_height = int(np.median(stixel_base))

            if stixel_base_height - stixel_top_height < min_stixel_height or stixel_top_height == 0:
                stixel_top_height = stixel_base_height - min_stixel_height

            stixel_median_disp = np.nanmedian(disparity_map[stixel_top_height:stixel_base_height, stixel_range])
            stixel_median_depth = np.nanmedian(depth_map[stixel_top_height:stixel_base_height, stixel_range])

            stixel = [stixel_top_height, stixel_base_height, stixel_median_disp, stixel_median_depth]
            self.rectangular_stixel_list.append(stixel)
            rectangular_stixel_mask[stixel_top_height:stixel_base_height, stixel_range] = 1

            cv2.imshow("Membership", normalized)
            cv2.imshow("normilized cost", normalized_cost)


        return self.rectangular_stixel_list, rectangular_stixel_mask


    def get_optimal_height(self, cost_image, depth_map, free_space_boundary, img_row_to_scanline=None):
        
        #cost_image = cv2.normalize(cost_image, None, 0, 255, cv2.NORM_MINMAX)
        height, width = cost_image.shape
        DP = np.full((height, width), np.inf, dtype=float)
        # Parent pointer: for each position, record the row index from previous column that gave the optimum.
        parent = -np.ones((height, width), dtype=int)
        NZ = 2 # 5
        Cs = 2 # 4 #8
        DP[:, 0] = cost_image[:, 0]

        def distance_transform_1d(f, penalty):
            n = len(f)
            dt = f.copy()
            argmin = np.arange(n)
            # Forward pass.
            for i in range(1, n):
                if dt[i-1] + penalty < dt[i]:
                    dt[i] = dt[i-1] + penalty
                    argmin[i] = argmin[i-1]
            # Backward pass.
            for i in range(n - 2, -1, -1):
                if dt[i+1] + penalty < dt[i]:
                    dt[i] = dt[i+1] + penalty
                    argmin[i] = argmin[i+1]
            return dt, argmin

        for u in range(width - 1):
            
            v_f_img  = int(free_space_boundary[u])
            v_f1_img = int(free_space_boundary[u+1])
            if img_row_to_scanline is not None:
                v_f = img_row_to_scanline[v_f_img]
                v_f1 = img_row_to_scanline[v_f1_img]
            else:
                v_f = v_f_img
                v_f1 = v_f1_img
            z_u   = depth_map[v_f, u]
            z_u1  = depth_map[v_f1, u+1]
            relax_factor = max(0, 1 - abs(z_u - z_u1) / NZ)
            # The penalty for a jump of 1 row.
            penalty = Cs * relax_factor
            dt, argmin = distance_transform_1d(DP[:, u], penalty)
            
            # Update the DP for column u+1.
            DP[:, u+1] = dt + cost_image[:, u+1]
            parent[:, u+1] = argmin

        # Identify the best ending row in the last column.
        best_end_v = np.argmin(DP[:, width - 1])

        # Backtrack to recover the boundary.
        boundary = np.empty(width, dtype=int)
        boundary[-1] = best_end_v
        for u in range(width - 1, 0, -1):
            boundary[u - 1] = parent[boundary[u], u]
        
        # Optionally, create a binary mask of the boundary.
        boundary_mask = np.zeros((height, width), dtype=int)
        boundary_mask[boundary, np.arange(width)] = 1

        return boundary, boundary_mask

    
    def get_greedy_height(self, cost_image):

        height, width = cost_image.shape
        boundary = np.zeros(width, dtype=np.int32)
        boundary_mask = np.zeros((height, width), dtype=int)    

        for u in range(width):
            # Find the row with the max cost in column u
            best_row = np.argmin(cost_image[:, u])
            boundary[u] = best_row
            boundary_mask[best_row, u] = 1

        return boundary, boundary_mask


    def get_stixel_3d_points(self, camera_params):
        stixel_list = self.rectangular_stixel_list
        stixel_3d_points = np.zeros((self.num_stixels, 4, 3))
        stixel_image_points = np.zeros((4, 2))
        for n, stixel in enumerate(stixel_list):
            top_height = stixel[0]
            base_height = stixel[1]
            left_bound = n * self.stixel_width
            right_bound = (n + 1) * self.stixel_width
            stixel_depth = self.fused_stixel_depth_list[n] # stixel[3]
            if np.isnan(stixel_depth):
                stixel_3d_points[n] = np.nan
                continue

            stixel_image_points[0] = [left_bound, top_height]
            stixel_image_points[1] = [right_bound, top_height]
            stixel_image_points[2] = [right_bound, base_height]
            stixel_image_points[3] = [left_bound, base_height]
            
            depth = np.array([stixel_depth, stixel_depth, stixel_depth, stixel_depth])

            stixel_3d_points[n] = ut.calculate_3d_points(stixel_image_points[:, 0], stixel_image_points[:, 1], depth, camera_params)

        return stixel_3d_points

            
    
    def filter_lidar_points_by_stixels(self, lidar_image_points, lidar_3d_points):

        filtered_image_points = []
        filtered_3d_points = []
        stixel_indices = []
        stixel_list = self.rectangular_stixel_list

        for n, stixel in enumerate(stixel_list):
            top_height = stixel[0]
            base_height = stixel[1]
            left_bound = n * self.stixel_width
            right_bound = (n + 1) * self.stixel_width

            # Check if points fall within this stixel's bounds (horizontal and vertical in pixel coordinates)
            mask = (
                (lidar_image_points[:, 1] >= top_height) &
                (lidar_image_points[:, 1] <= base_height) &
                (lidar_image_points[:, 0] >= left_bound) &
                (lidar_image_points[:, 0] <= right_bound)
            )

            stixel_points_2d = lidar_image_points[mask]
            stixel_points_3d = lidar_3d_points[mask]  # Filter corresponding 3D points

            filtered_image_points.extend(stixel_points_2d)
            filtered_3d_points.extend(stixel_points_3d)
            stixel_indices.extend([n] * len(stixel_points_2d))

        filtered_image_points = np.array(filtered_image_points)
        filtered_3d_points = np.array(filtered_3d_points)
        stixel_indices = np.array(stixel_indices)

        return filtered_image_points, filtered_3d_points, stixel_indices
    

    def get_stixel_depth_from_lidar_points(self, lidar_3d_points, stixel_indices):
        stixel_depths = []
        for n in range(self.num_stixels):
            mask = stixel_indices == n
            stixel_points = lidar_3d_points[mask]
            
            if len(stixel_points) > 0:
                distances = stixel_points[:, 2]
                #distances = np.linalg.norm(stixel_points, axis=1)
                stixel_depth = np.nanmedian(distances)  # Take the median distance for robustness
            else:
                stixel_depth = np.nan  # Assign NaN or a placeholder for empty stixels
            stixel_depths.append(stixel_depth)
        return np.array(stixel_depths)
    
    
    def calculate_2d_points_from_stixel_positions(stixel_positions, stixel_width, depth_map, cam_params):
        stixel_positions = stixel_positions[stixel_positions[:, 0].argsort()]

        d = np.array([])    
        d_invalid = np.array([])
        for n, stixel_pos in enumerate(stixel_positions):
            x_start = max(0, n * stixel_width - stixel_width // 2)
            x_end = min(depth_map.shape[1], (n + 1) * stixel_width - stixel_width // 2)
            depth_along_stixel = depth_map[int(stixel_pos[1]), x_start:x_end]
            depth_along_stixel = depth_along_stixel[depth_along_stixel > 0]
            depth_along_stixel = depth_along_stixel[~np.isnan(depth_along_stixel)]
            if depth_along_stixel.size == 0: #no depth values in stixel
                #stixel_positions = np.delete(stixel_positions, n, axis=0)
                d_invalid = np.append(d_invalid, int(n))
            else:
                median_depth = np.median(depth_along_stixel[depth_along_stixel > 0])
                #avg_depth = np.mean(depth_along_stixel[depth_along_stixel > 0])
                d = np.append(d, median_depth)

        
        d_invalid = np.array(d_invalid, dtype=int)
        X = stixel_positions[:, 0]
        X = np.delete(X, d_invalid)
        Y = stixel_positions[:, 1]
        Y = np.delete(Y, d_invalid)
        points_3d = ut.calculate_3d_points(X, Y, d, cam_params)
        points_2d = points_3d[:, [0, 2]]

        return points_2d
    
    def get_polygon_points_from_lidar_and_stereo_depth(self, lidar_stixel_depths, stixel_positions, cam_params, sigma_px=1, sigma_z_lidar=0.1):

        Z = np.full(len(self.rectangular_stixel_list), np.nan)
        Z_invalid = np.array([], dtype=int)

        for n, stixel in enumerate(self.rectangular_stixel_list):
            px = stixel[2]
            z_stereo = stixel[3]
            z_lidar = lidar_stixel_depths[n]
            #print(z_lidar)

            #if np.isnan(z_stereo) and np.isnan(z_lidar):
            if np.isnan(z_lidar) or z_lidar == 0:
                #print(1)
                #Z_invalid = np.append(Z_invalid, n)
                prev_z = self.prev_rectangular_stixel_list[n][3]
                Z[n] = prev_z #np.nan

            elif np.isnan(z_stereo) or px == 0:
                z_fused = z_lidar
                Z[n] = z_fused
                #print(2)
            elif np.isnan(z_lidar) or z_lidar == 0:
                z_fused = z_stereo
                #print(3)
                Z[n] = z_fused
            else:
                sigma_z_stereo = sigma_px * z_stereo / px
                sigma_z_squared = 1 / (1 / sigma_z_stereo**2 + 1 / sigma_z_lidar**2)  # Combine stereo and lidar depth uncertainties
                z_fused = sigma_z_squared * (z_lidar / sigma_z_lidar**2 + px / sigma_z_stereo**2)
                #print(4)
                #Z[n] = z_fused
                Z[n] = z_lidar

            self.rectangular_stixel_list[n][3] = Z[n]
            
            #print(f"z_fused: {z_fused}")
            #if Z[n] > 40:
             #   Z[n] = 40
            
        self.fused_stixel_depth_list = Z

        X = stixel_positions[:, 0]
        Y = stixel_positions[:, 1]
        X = np.delete(X, Z_invalid)
        Y = np.delete(Y, Z_invalid)
        Z = np.delete(Z, Z_invalid)
        
        points_3d = ut.calculate_3d_points(X, Y, Z, cam_params)
        points_2d = points_3d[:, [0, 2]]

        # Compute the angle of each point relative to the origin (0,0)
        angles = np.arctan2(points_2d[:, 1], points_2d[:, 0])

        # Sort points by angle to create a continuous polygon boundary
        sorted_indices = np.argsort(angles)
        points_2d_sorted = points_2d[sorted_indices]

        return points_2d_sorted
    

    def merge_stixels_onto_image(self, stixel_list, image):

        overlay = np.zeros_like(image)
        stixel_width = self.get_stixel_width(image.shape[1])
        disp_values = [stixel[2] for stixel in stixel_list]
        min_disp = 0 #np.min(disp_values)
        max_disp = 10 #np.max(disp_values)
    

        for n, stixel in enumerate(stixel_list):
            stixel_top = stixel[0]
            stixel_base = stixel[1]
            stixel_disp = stixel[2]

            #print(f"Stixel {n}: top={stixel_top}, base={stixel_base}, width={stixel_width}")

            if stixel_base > stixel_top and stixel_width > 0:

                #normalized_disp = np.uint8(255 * (stixel_disp - min_disp) / (max_disp - min_disp))
                #normalized_disp_array = np.full((stixel_base - stixel_top, stixel_width), normalized_disp, dtype=np.uint8)
                #colored_stixel = cv2.applyColorMap(normalized_disp_array, cv2.COLORMAP_JET)
                green_stixel = np.full((stixel_base - stixel_top, stixel_width, 3), (0, 80, 0), dtype=np.uint8) #(0, 50, 0)

                overlay[stixel_top:stixel_base, n * stixel_width:(n + 1) * stixel_width] = green_stixel

                # Add a border (rectangle) around the stixel
                cv2.rectangle(overlay, 
                        (n * stixel_width, stixel_top),  # Top-left corner
                        ((n + 1) * stixel_width, stixel_base),  # Bottom-right corner
                        (0,0,0),  # Color of the border (BGR)
                        2)  # Thickness of the border

        alpha = 0.8  # Weight of the original image
        beta = 1  # Weight of the overlay
        gamma = 0.0  # Scalar added to each sum

        blended_image = cv2.addWeighted(image, alpha, overlay, beta, gamma)
        return blended_image
        

    
def create_polygon_from_2d_points(points: list) -> Polygon:

    if len(points) < 2:
        print("Cannot create a polygon with less than 2 points.")
        return Polygon()
    #sorted_indices = points[:, 0].argsort()
    #points = points[sorted_indices]
    origin = np.array([0, 0])
    polygon_points = np.vstack([origin, points])

    return Polygon(polygon_points)