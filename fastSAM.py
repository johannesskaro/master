import numpy as np
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor
import cv2
import os

class FastSAMSeg:
    """
    A class to handle FastSAM segmentation tasks.
    """

    def __init__(self, model_path: str ='./weights/FastSAM-x.pt'):
        """
        Initialize the FastSAMSeg class.

        Parameters:
        - model_path (str): Path to the pretrained FastSAM model.
        """
        try:
            self.model = FastSAM(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading FastSAM model from {model_path}. Reason: {e}")
        
    def _segment_img(self, img: np.array, device: str = 'mps') -> np.array:
        """
        Internal method to perform segmentation on the provided image.

        Parameters:
        - img (np.array): Input image for segmentation.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Segmentation results.
        """
        retina_masks = True
        verbose = False
        results = self.model(img, device=device, retina_masks=retina_masks, verbose=verbose)
        return results

    def get_mask_at_points(self, img: np.array, points: np.array, pointlabel: np.array, device: str = 'mps') -> np.array:
        """
        Obtain masks for specific points on the image.

        Parameters:
        - img (np.array): Input image.
        - points (np.array): Array of points.
        - pointlabel (np.array): Corresponding labels for points.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Mask result.
        """
        mask = np.zeros((img.shape[0], img.shape[1]))
        point_results = self.model(img, points=points, labels=pointlabel, device=device, retina_masks=True, verbose=False)
        ann = point_results[0].cpu()
        if len(ann)>0:
            mask = np.array(ann[0].masks.data[0]) if ann[0].masks else np.zeros((img.shape[0], img.shape[1]))

        return mask
    
    def get_mask_at_bbox(self, img: np.array, bbox: np.array, device: str = 'mps') -> np.array:
        """
        Obtain masks for the bounding box on the image.

        Parameters:
        - img (np.array): Input image.
        - bbox (np.array): Bounding box.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Mask result.
        """
        box_results = self.model(img, bboxes=bbox, device=device, retina_masks=True)
        ann = box_results[0].cpu()
        mask = np.array(ann[0].masks.data[0]) if ann[0].masks else np.zeros((img.shape[0], img.shape[1]))

        return mask
    
    def get_all_masks(self, img: np.array, device: str = 'cuda') -> np.array:
        """
        Obtain all masks for the input image.

        Parameters:
        - img (np.array): Input image.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Masks result.
        """

        results = self._segment_img(img, device=device)
        if results[0].masks is not None:
            masks = np.array(results[0].masks.data.cpu())
        else: 
            masks = np.array([])
        return masks
    
    def get_all_countours(self, img: np.array, device: str = 'cuda', min_area=5000) -> np.array:
        """
        Obtain all contours for the input image.

        Parameters:
        - img (np.array): Input image.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Contours result.
        """
        masks = self.get_all_masks(img, device=device)
        contour_mask = np.zeros((img.shape[0], img.shape[1]))

        for mask in masks:
            mask = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=1)

        return contour_mask
    
    def get_all_upper_countours(self, img: np.array, device: str = 'cuda', min_area=3000) -> np.array:
        masks = self.get_all_masks(img, device=device)
        combined_upper = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for mask in masks:
            # Ensure binary mask
            bin_mask = (mask > 0).astype(np.uint8)
            # Optionally, you can use cv2.findContours to filter out small regions
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = False
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    valid = True
                    break
            if not valid:
                continue

            # Extract the upper contour line from this mask.
            upper_line = self.get_upper_contour_line(bin_mask)
            # Optionally, you can also draw the original contour:
            # cv2.drawContours(upper_line, [contour], -1, 255, thickness=1)

            # Combine the upper contour line with previous ones.
            combined_upper = cv2.bitwise_or(combined_upper, upper_line)

        return combined_upper
    
    def get_upper_contour_line(self, mask) -> np.array:
        has_nonzero = mask.any(axis=0)  # shape: (W,)
        
        # For each column, np.argmax returns the index of the first occurrence of the maximum value.
        # For a binary mask (0 or 1), this gives the first nonzero pixel. Note that for columns
        # with all zeros, np.argmax returns 0 even though no pixel is nonzero.
        first_nonzero_indices = np.argmax(mask, axis=0)  # shape: (W,)
        
        # Create the result mask (initialize with zeros)
        result_mask = np.zeros_like(mask, dtype=np.uint8)
        
        # Only update the columns that have at least one nonzero pixel.
        valid_cols = np.where(has_nonzero)[0]
        result_mask[first_nonzero_indices[valid_cols], valid_cols] = 255

        return result_mask
    
    def get_bottom_contours(self, contour_mask) -> np.array:
        height, width = contour_mask.shape
        result_mask = np.zeros_like(contour_mask, dtype=np.uint8)

        reversed_mask = (contour_mask[::-1, :] > 0)

        first_idx = np.argmax(reversed_mask, axis=0)
        has_nonzero = np.any(reversed_mask, axis=0)
        row_indices = np.where(has_nonzero, height - 1 - first_idx, -1)
        cols = np.arange(width)
        valid = row_indices >= 0
        result_mask[row_indices[valid], cols[valid]] = 1
        return result_mask

    
    def get_horizontal_contours(self, img: np.array, device: str = 'cuda', min_area=3000, angle_threshold=10) -> np.array:

        masks = self.get_all_masks(img, device=device)
        H, W = img.shape[1], img.shape[2]
        horizontal_mask = np.zeros((H, W), dtype=np.uint8)

        for mask in masks:
            mask_bin = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue

                # Approximate contour to reduce the number of points
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Iterate through line segments in the approximated contour
                for i in range(len(approx)):
                    pt1 = approx[i][0]
                    pt2 = approx[(i + 1) % len(approx)][0]  # Wrap-around to the first point

                    # Compute the difference in x and y
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]

                    if dx == 0:
                        continue

                    # Calculate the angle in degrees
                    angle = np.degrees(np.arctan2(dy, dx))

                    # Normalize angle to the range [0, 180)
                    angle = abs(angle) % 180

                    # Check if the line is near-horizontal
                    if angle <= angle_threshold or angle >= (180 - angle_threshold):
                        cv2.line(horizontal_mask, tuple(pt1), tuple(pt2), 255, thickness=1)

        return horizontal_mask
    
    