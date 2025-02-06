import numpy as np
from shapely.geometry import Polygon
from collections import deque
import utilities as ut
import cv2

class Stixels:

    N = 10
    stixel_2d_points_N_frames = deque(maxlen=N)
    rectangular_stixel_list = []
    fused_stixel_depth_list = []

    def __init__(self, num_of_stixels = 96) -> None:
        self.num_stixels = num_of_stixels # 958//47 #gir 20I wan 

    def get_stixel_2d_points_N_frames(self) -> np.array:
        return self.stixel_2d_points_N_frames

    def add_stixel_2d_points(self, stixel_2d_points: np.array) -> None:
        self.stixel_2d_points_N_frames.append(stixel_2d_points)

    def get_stixel_width(self, img_width) -> int:
        self.stixel_width = int(img_width // self.num_stixels)
        return self.stixel_width

    def get_free_space_boundary(self, water_mask: np.array) -> np.array:

        height, width = water_mask.shape

        free_space_boundary_mask = np.zeros_like(water_mask)
        #free_space_boundary = np.zeros(width)
        free_space_boundary = np.ones(width) * height

        for j in range(width):
            for i in reversed(range(height-50)):
                if water_mask[i, j] == 0:
                    free_space_boundary_mask[i, j] = 1
                    free_space_boundary[j] = i
                    break

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
        offset = 0.3
        min_stixel_height = 20

        rectangular_stixel_mask = np.zeros_like(water_mask)
        self.rectangular_stixel_list = []

        for n in range(self.num_stixels):
            stixel_range = slice(n * stixel_width, (n + 1) * stixel_width)
            stixel_base = free_space_boundary[stixel_range]
            stixel_base_height = int(np.median(stixel_base))
            stixel_top_height = stixel_base_height - min_stixel_height

            # Forhåndsberegn rad-medianer for de aktuelle kolonnene (hele bildet)
            stixel_disparity = disparity_map[:, stixel_range]
            row_medians = np.nanmedian(stixel_disparity, axis=1)

            stixel_depth = depth_map[stixel_base_height, stixel_range]
            base_depth = np.nanmedian(stixel_depth)

            adaptive_threshold = std_dev_threshold_base * (base_depth / ref_depth) + offset

            # Inkrementell standardavvik 
            mean = 0.0
            M2 = 0.0
            count = 0

            # Gå nedover fra base_height til 0
            for v in range(stixel_base_height, -1, -1):
                x = row_medians[v]
                count += 1

                # Oppdater løpende mean og M2
                delta = x - mean
                mean += delta / count
                delta2 = x - mean
                M2 += delta * delta2

                if count > 1:
                    current_std = np.sqrt(M2 / (count - 1))
                    if current_std > adaptive_threshold:
                        stixel_top_height = v
                        if (stixel_base_height - v) < min_stixel_height:
                            stixel_top_height = stixel_base_height - min_stixel_height
                        break

            # Median av disparity- og depth-region
            stixel_median_disp = np.nanmedian(disparity_map[stixel_top_height:stixel_base_height, stixel_range])
            stixel_median_depth = np.nanmedian(depth_map[stixel_top_height:stixel_base_height, stixel_range])

            stixel = [stixel_top_height, stixel_base_height, stixel_median_disp, stixel_median_depth]
            self.rectangular_stixel_list.append(stixel)

            rectangular_stixel_mask[stixel_top_height:stixel_base_height, stixel_range] = 1

        return self.rectangular_stixel_list, rectangular_stixel_mask
    

    def smooth_stixel_tops_by_depth(self, disparity_map, depth_map, depth_threshold_rel=0.1):

        num_stixels = self.num_stixels

        original_tops = np.array([stixel[0] for stixel in self.rectangular_stixel_list])
        median_depths = np.array([stixel[3] for stixel in self.rectangular_stixel_list])
        
        # Initialize the smoothed top positions as a copy of the original.
        smoothed_tops = original_tops.copy()
        
        # For each stixel, consider a neighborhood (here, the previous and next stixels).
        for i in range(num_stixels):
            neighbor_indices = []
            for j in range(max(0, i - 1), min(num_stixels, i + 2)):
                # Check if the neighbor stixel's median depth is similar.
                # Here we use a relative threshold.
                if abs(median_depths[i] - median_depths[j]) < depth_threshold_rel * median_depths[i]:
                    neighbor_indices.append(j)
                    
            # If we have any valid neighbors (including the stixel itself), take the median.
            if neighbor_indices:
                smoothed_tops[i] = int(np.median(original_tops[neighbor_indices]))
        
        # Optionally, update the stixel list and the corresponding mask.
        stixel_width = self.get_stixel_width(disparity_map.shape[1])
        rectangular_stixel_mask = np.zeros_like(disparity_map)
        for n in range(num_stixels):
            stixel_range = slice(n * stixel_width, (n + 1) * stixel_width)
            # The base positions remain unchanged.F
            stixel_base = self.rectangular_stixel_list[n][1]
            stixel_top = smoothed_tops[n]

            
            # Update the stixel info (optionally recompute median disparity/depth)
            stixel_median_disp = np.nanmedian(disparity_map[stixel_top:stixel_base, stixel_range])
            stixel_median_depth = np.nanmedian(depth_map[stixel_top:stixel_base, stixel_range])
            self.rectangular_stixel_list[n] = [stixel_top, stixel_base, stixel_median_disp, stixel_median_depth]
            
            rectangular_stixel_mask[stixel_top:stixel_base, stixel_range] = 1
            
        # Store or return the updated stixel list and mask.
        self.rectangular_stixel_mask = rectangular_stixel_mask
        return self.rectangular_stixel_list, rectangular_stixel_mask

    

    def create_stixels(self, disparity_map, depth_map, free_space_boundary, cam_params):
        height, width = disparity_map.shape
        b = cam_params["b"]
        fx = cam_params["fx"]
        Delta_Z = 2 #2
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

        return self.rectangular_stixel_list, rectangular_stixel_mask, normalized, normalized_cost, boundary_mask, boundary_mask_greedy

    
    def create_membership_image(self, disparity_map, depth_map, free_space_boundary, cam_params):
        height, width = disparity_map.shape
        b = cam_params["b"]
        fx = cam_params["fx"]
        Delta_Z = 2 #2
        membership_image = np.zeros((height, width))

        for u in range(width):
            v_f = int(free_space_boundary[u])
            d_hat = disparity_map[v_f, u]
            z_u = depth_map[v_f, u]
            Delta_D = d_hat - (fx * b) / (z_u + Delta_Z)

            for v in range(height):
                d_uv = disparity_map[v, u]
                exponent = 1 - ((d_uv - d_hat) / Delta_D)**2
                membership_image[v, u] = 2**exponent - 1


        return membership_image


    def create_cost_image(self, membership_image, disparity_map, free_space_boundary):
        height, width = disparity_map.shape

        cost_image = np.zeros((height, width))
        
        for u in range(width):
            v_f = int(free_space_boundary[u])
            prefix = np.zeros(v_f + 2, dtype=np.float32)
            

            for v in range(v_f + 1):
                prefix[v + 1] = prefix[v] + membership_image[v, u]

            for v in range(v_f + 1):
                sum_above = prefix[v]

                sum_below = prefix[v_f + 1] - prefix[v]
        
                cost_image[v, u] = sum_above - sum_below

        return cost_image


    def get_optimal_height(self, cost_image, depth_map, free_space_boundary):

        height, width = cost_image.shape
        
        # DP[v, u]: best cost to reach row v at column u.
        DP = np.full((height, width), np.inf, dtype=float)
        # Parent pointer: for each position, record the row index from previous column that gave the optimum.
        parent = -np.ones((height, width), dtype=int)
        
        # Parameters (example values)
        NZ = 5
        Cs = 8

        # Initialize the first column.
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

        # Process each column transition.
        for u in range(width - 1):
            # Compute relax_factor from depth differences at the free-space boundary.
            v_f  = int(free_space_boundary[u])
            v_f1 = int(free_space_boundary[u+1])
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

    

    def get_optimal_height_2(self, cost_image, depth_map, free_space_boundary):
        height, width = cost_image.shape
        #    DP[i,j] = min cost of path that chooses row j in column i
        DP = np.full((height, width), np.inf, dtype=float)
        # track parent to reconstruct best boundary
        parent = -1 * np.ones((height, width), dtype=int)
        NZ = 5
        Cs = 8

        for v in range(height):
            DP[v, 0] = cost_image[v, 0]

        max_jump = 5
        # Forward pass
        for u in range(width - 1):
            v_f = int(free_space_boundary[u])
            v_f1  = int(free_space_boundary[u+1])
            z_u   = depth_map[v_f, u]
            z_u1  = depth_map[v_f1, u+1]
            depth_diff = abs(z_u - z_u1)
            relax_factor = max(0, 1 - depth_diff / NZ)  # (1 - |z_i - z_{i+1}| / N_Z)

            for v0 in range(height):
                lower = max(0, v0 - max_jump)
                upper = min(height-1, v0 + max_jump)

                for v1 in range(lower, upper+1):
                    jump = abs(v0 - v1)
                    Sij = Cs * jump * relax_factor

                    edge_cost = cost_image[v1, u+1] + Sij
                    new_cost = DP[v0, u] + edge_cost


                    if new_cost > DP[v1, u+1]:
                        DP[v1, u+1]    = new_cost
                        parent[v1, u+1] = v0

        best_end_v = np.argmin(DP[:,width-1])
        best_cost  = DP[best_end_v, width-1]

        # Backtrack to get the path of row indices
        boundary = np.zeros(width, dtype=int)
        boundary_mask = np.zeros((height, width), dtype=int)
        boundary[width-1] = best_end_v

        for u in reversed(range(width-1)):
            boundary[u] = parent[boundary[u+1], u+1]
        
        for u in range(width):
            v = boundary[u]
            boundary_mask[v, u] = 1

        

        return boundary, boundary_mask, best_cost
    
    def get_greedy_height(self, cost_image, free_space_boundary):

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

            if np.isnan(z_stereo) and np.isnan(z_lidar):
                #print(1)
                Z_invalid = np.append(Z_invalid, n)

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
                Z[n] = z_fused
            
            #print(f"z_fused: {z_fused}")
            
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