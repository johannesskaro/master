import cv2
import numpy as np


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

def create_cost_map(self, disparity_map, depth_map, free_space_boundary, min_stixel_height):
        height, width = disparity_map.shape
    
        normalized_disparity = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
        normalized_disparity = normalized_disparity.astype(np.uint8)
        blurred_image = cv2.GaussianBlur(normalized_disparity, (5, 5), 0)
        grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        grad_y = cv2.convertScaleAbs(grad_y)
        _, grad_y = cv2.threshold(grad_y, 75, 255, cv2.THRESH_BINARY)
        grad_y = grad_y / 255

        cost_map = np.full((height, self.num_stixels), 255, dtype=float)
        free_space_boundary_depth = np.zeros((height, self.num_stixels))

        prev_stixels = self.rectangular_stixel_list

        for n in range(self.num_stixels):
            stixel_range = slice(n * self.stixel_width, (n + 1) * self.stixel_width)
            stixel_base = free_space_boundary[stixel_range]
            v_f = int(np.median(stixel_base))

            stixel_disparity = disparity_map[:, stixel_range]
            row_medians = np.nanmedian(stixel_disparity, axis=1)
            depth_window = depth_map[v_f-10:v_f+1, stixel_range]
            free_space_boundary_depth[v_f, n] = np.nanmedian(depth_window)

            grad_y[v_f-10:v_f+1, stixel_range] = 0
            prev_v_top = prev_stixels[n][0]
            prev_depth = prev_stixels[n][3]

            mean = 0.0
            M2 = 0.0
            count = 0
            current_std = 0.0

            w1 = 200 #250 #250 #255
            w2 = 200 #175 #150
            w3 = 0 #175 #100
            w4 = 150 #150
            w5 = 0 #1
            w6 = 0

            local_window = 5
            local_std = 0

            d_hat = row_medians[v_f]

            Delta_D = 1 # 0.6
 
            for v in range(v_f, -1, -1):
                x = row_medians[v]
                count += 1
                delta = x - mean
                mean += delta / count
                delta2 = x - mean
                M2 += delta * delta2
                
                if prev_v_top == 0:
                    delta_height = 0
                    height_cost = 0
                else:
                    delta_height = abs(v - prev_v_top)
                    scale = max(0, 1 - (abs(prev_depth - free_space_boundary_depth[v_f, n]) / 5))
                    height_cost = delta_height * scale

                exponent = 1 - ((x - d_hat) / Delta_D)**2
                membership_cost = 2**exponent - 1

                window_start = max(0, v - local_window + 1)
                prev_local_std = local_std
                local_std = np.nanstd(row_medians[window_start:v+1])
                
                local_std_threshold = 0.15 # 0.15
                if local_std > local_std_threshold and count > 10:
                    high_local_std = 1
                else:
                    high_local_std = 0

                if count > 1:
                    current_std = np.sqrt(M2 / (count - 1))
                else:
                    current_std = 0

                if current_std > 0.35: # and first_obstacle == 1:
                    high_std = 1
                else:
                    high_std = 0

                grad_y_v = np.mean(grad_y[v, stixel_range])

                std_dev_cost = 2**(1 - 2*(current_std**2))
                
                cost_map[v, n] = - w1 * grad_y_v - w2 * std_dev_cost + w3 * high_std - w4 * high_local_std + w5 * height_cost - w6 * membership_cost


        cost_map = cv2.normalize(cost_map, None, 0, 255, cv2.NORM_MINMAX)
        cost_map = cost_map.astype(np.uint8)


        new_width = width * 1  # Make rows thicker
        cost_map_resized = cv2.resize(-cost_map, (new_width, height), interpolation=cv2.INTER_NEAREST)
        #cost_map_colored = cv2.applyColorMap(cost_map_resized, cv2.COLORMAP_JET)
        cv2.imshow("Cost Map", cost_map_resized)
        #cv2.imshow("grad_y", grad_y)
        #colored_disparity = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET)
        #cv2.imshow("colored_disparity", colored_disparity)

        return cost_map, free_space_boundary_depth


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