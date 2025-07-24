from depth_utils import pull_data

import os
import cv2
import numpy as np
from tqdm import tqdm


def shift_data(data:np.ndarray, num_pixels:int=16):

    #shift data left x number of pixels to match to correct for offset caused by baseline of stereo cameras.
    shift = num_pixels
    temp = data[:, :, shift:].copy()
    data[:, :, :-shift] = temp
    data[:, :, -shift:] = np.nan  # fill the last shift pixels with NaN
    return data


def mask_horizon(raw_image:np.ndarray, canny_thres1:int=170,
                 canny_thres2:int=9700, hough_thres:int=100,
                 min_line_length:int=100, max_line_gap:int=10):

    # Use canny edge detection and hough line transform to find best line to match horizon
    canny = cv2.Canny(image=raw_image, threshold1=canny_thres1, threshold2=canny_thres2, apertureSize=7, L2gradient=False)
    lines = cv2.HoughLinesP(canny, rho=1, theta=np.pi/180, threshold=hough_thres, minLineLength=min_line_length, maxLineGap=max_line_gap)

    # collect possible horizon candidates
    horizon_candidates = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if abs(angle) < 10:  # if angle is close to horizontal
                horizon_candidates.append(line[0])  #add line to candidates

    # select top most line and extend it to the borders of the image
    if horizon_candidates:
        top_line = min(horizon_candidates, key=lambda line: min(line[1], line[3]))  #find the line with the highest y coordinates. Topmost line
        x1, y1, x2, y2 = top_line
        # Extend the line to the borders of the image
        height, width = raw_image.shape[:2]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        xstart = 0
        xend = width - 1
        ystart = int(m * xstart + b)   #used for plotting the horizon line
        yend = int(m * xend + b)
        horizon = (xstart, xend, ystart, yend)

        # mask values above the line - loop through pixel x coords
        mask = np.zeros_like(raw_image, dtype=np.uint8)
        for x in range(width):
            y = int(m * x + b)
            if 0 <= y < height:
                if mask.ndim == 2:  # grayscale image
                    mask[:y, x] = 255
                else:  # color image
                    mask[:y, x] = [255, 255, 255]

        return mask, horizon


def load_data(cam:str, dataset:str, data_folder:str):

    # Load data using flag for determining which sensor and which dataset
    # cam can be RGB or NIR
    # dataset TODO: number the datasets, dataset 0, 1, etc... Using surf and else is too focused

    if cam == 'RGB':

        if dataset == 'surf':

            rgb_mde = np.load(f'{data_folder}/ms2_surf_aux_image_rect_color_mde.npy', allow_pickle=True).item() #RGB mde depth
            left_depth = pull_data('/mnt/e/ms_output_2/ms2_surf/left_depth/data.npz') #stereo depth
            aux_color_rect = pull_data('/mnt/e/ms_output_2/ms2_surf/aux_image_rect_color/data.npz') #rgb image
            dict_list = [left_depth, rgb_mde, aux_color_rect]
            return dict_list

        else:

            rgb_mde = np.load(f'{data_folder}/ms2_aux_image_rect_color_mde.npy', allow_pickle=True).item()
            left_depth = pull_data('/mnt/e/ms_output/ms2/left_depth/data.npz')
            aux_color_rect = pull_data('/mnt/e/ms_output/ms2/aux_image_rect_color/data.npz')
            dict_list = [left_depth, rgb_mde, aux_color_rect]
            return dict_list

    if cam == 'NIR':

        if dataset == 'surf':
            nir_mde = np.load(f'{data_folder}/ms1_surf_left_image_rect_mde.npy', allow_pickle=True).item()
            left_depth = pull_data('/mnt/e/ms_output_2/ms1_surf/left_depth/data.npz')
            nir_left = pull_data('/mnt/e/ms_output_2/ms1_surf/left_image_rect/data.npz')  # rectified left IR
            dict_list = [left_depth, nir_mde, nir_left]
            return dict_list

        else:

            nir_mde = np.load(f'{data_folder}/ms1_left_image_rect_mde.npy', allow_pickle=True).item()
            left_depth = pull_data('/mnt/e/ms_output/ms1/left_depth/data.npz')
            nir_left = pull_data('/mnt/e/ms_output/ms1/left_image_rect/data.npz')  # rectified left IR
            dict_list = [left_depth, nir_mde, nir_left]
            return dict_list


def clip_data(images, mde_depths, stereo_depths, flag:str):

    #takes arrays and returns clipped versions of them. Flag is either 'RGB' or 'NIR'
    #make copies of arrays for clipping and calculations
    images_ = np.array(images).copy()  # RGB or NIR images
    mde_depths_ = np.array(mde_depths).copy()  # MDE depth maps
    stereo_depths_ = np.array(stereo_depths).copy()  # Stereo depth maps

    #find horizon and create mask
    masks = []
    horizons = []
    for i in range(len(images_)):
        mask, horizon = mask_horizon(images_[i])
        masks.append(mask)
        horizons.append(horizon)
    masks = np.array(masks)
    horizons = np.array(horizons)

    if flag == 'RGB':
        # MDE prediction is run off of RGB image. Stereo is calculated from monochrome cams. Offset in camera location, must shift to correct for it.
        stereo_depths = shift_data(stereo_depths_, num_pixels=16) # Shift stereo depth map to match MDE depth map

    # apply mask to depth maps and background image
    if masks.ndim > 3:
        masks = masks[:, :, :, 0]  # Remove the channel dimension if it exists
    for i in range(len(mde_depths_)):
        mde_depths_[i] = np.where(masks[i] != 255, mde_depths_[i], np.nan)
        stereo_depths_[i] = np.where(masks[i] != 255, stereo_depths_[i], np.nan)

    # create box around the LARC, create mask of NaNs where the LARC is
    yy, xx = np.meshgrid(np.arange(mde_depths_.shape[1]), np.arange(mde_depths_.shape[2]), indexing='ij')  # RGB and NIR images should be the same size

    if flag == 'RGB':
        larc_mask = ((xx >= 500) & (yy >= 530)) | (xx >= 800)  # creates 600x960 mask
        larc_mask = np.broadcast_to(larc_mask, mde_depths_.shape)  # broadcast 2d larc_mask to shape of all depth maps nx600x960
        mde_depths_[larc_mask] = np.nan  # apply mask to MDE depth map
        stereo_depths_[larc_mask] = np.nan  # apply mask to Stereo depth map

    if flag =='NIR':
        larc_mask = ((xx >= 370) & (yy >= 400)) | ((620 <= xx) & (xx <= 720) & (yy >= 120))
        larc_mask = np.broadcast_to(larc_mask, mde_depths_.shape)  # broadcast 2d larc_mask to shape of all depth maps
        mde_depths_[larc_mask] = np.nan  # apply mask to MDE depth map
        stereo_depths_[larc_mask] = np.nan  # apply mask to Stereo depth map

    # mask out values greater than or equal to 80m
    mde_depths_[mde_depths_ >= 80] = np.nan
    stereo_depths_[stereo_depths_ >= 80] = np.nan

    return mde_depths_, stereo_depths_, horizons


def read_dict(flag:str, dicts:dict):

    if flag == 'RGB':
        images = dicts['aux_image_rect_color']['messages']  # RGB Image
        mde_depths = dicts['aux_image_rect_color_mde']['messages']  # RGB MDE Depth
        stereo_depths = dicts['left_depth']['messages']  # StereoDepth
        return images, mde_depths, stereo_depths

    elif flag == 'NIR':
        images = dicts['left_image_rect']['messages']  # NIR Image
        mde_depths = dicts['left_image_rect_mde']['messages']  # NIR MDE Depth
        stereo_depths = dicts['left_depth']['messages']  # StereoDepth
        return images, mde_depths, stereo_depths

    elif flag == 'BOTH':
        rgb_images = dicts['aux_image_rect_color_rgb']['messages']  # RGB image
        nir_images = dicts['left_image_rect']['messages'] # NIR image
        rgb_mde_depths = dicts['aux_image_rect_color_mde_rgb']['messages'] # RGB MDE Depth
        nir_mde_depths = dicts['left_image_rect_mde']['messages']  # NIR MDE depth
        rgb_stereo_depths = dicts['left_depth_rgb']['messages']  # RGB StereoDepth
        nir_stereo_depths = dicts['left_depth']['messages']  # NIR StereoDepth
        return rgb_images, rgb_mde_depths, rgb_stereo_depths, nir_images, nir_mde_depths, nir_stereo_depths

    else:
        raise ValueError(f"Unknown flag: {flag}. Options are 'RGB', 'NIR', or 'BOTH'.")


def bin_data(stereo_depths_, mde_depths_, n_bins:int=81):

    num_bins = np.arange(0, n_bins, 1)
    stereo_depths_binned = np.digitize(stereo_depths_.flatten(), num_bins).reshape(stereo_depths_.shape)
    #calculate mean and std dev of MDE predictions in each bin
    mde_mean = []
    mde_std_dev = []
    X = []

    print('Processing...')
    for i in tqdm(range(len(num_bins))):
        mask = stereo_depths_binned == i
        if np.any(mask):
            mask = np.where(stereo_depths_binned == i, mde_depths_, np.nan)
            mde_mean.append(np.nanmean(mask, axis=(1, 2)))
            mde_std_dev.append(np.nanstd(mask, axis=(1, 2)))
            X.append(num_bins[i])

    return X, mde_mean, mde_std_dev


def cal_rmses(mde_depths_, stereo_depths_, abs_diff):

    #calculate rmse of difference map
    rmse = np.sqrt(np.nanmean((abs_diff**2), axis=(1, 2))) # Calculate RMSE for each depth map
    #calculate different RMSE values at different depths
    abs_diff_10m = np.where(abs_diff < 10, abs_diff, np.nan)  # Mask values greater than or equal to 10m
    rmse_10m = np.sqrt(np.nanmean((abs_diff_10m**2), axis=(1, 2)))  # Calculate RMSE for each depth map within 10m
    abs_diff_20m = np.where(abs_diff < 20, abs_diff, np.nan)  # Mask values greater than or equal to 20m
    rmse_20m = np.sqrt(np.nanmean((abs_diff_20m**2), axis=(1, 2)))  # Calculate RMSE for each depth map within 20m
    abs_diff_40m = np.where(abs_diff < 40, abs_diff, np.nan)  # Mask values greater than or equal to 40m
    rmse_40m = np.sqrt(np.nanmean((abs_diff_40m**2), axis=(1, 2)))  # Calculate RMSE for each depth map within 40m

    #RMSPE - average magnitude of error as a percentage of the acutal values
    stereo_depths_10m = np.where(stereo_depths_ < 10, stereo_depths_, np.nan)  # Mask values greater than or equal to 10m
    stereo_depths_20m = np.where(stereo_depths_ < 20, stereo_depths_, np.nan)  # Mask values greater than or equal to 20m
    stereo_depths_40m = np.where(stereo_depths_ < 40, stereo_depths_, np.nan)  # Mask values greater than or equal to 40m
    rmse_p = (np.sqrt(np.nanmean(((stereo_depths_ - mde_depths_)/stereo_depths_), axis=(1, 2))**2)) * 100  #RMSPE formula
    rmse_10p = (np.sqrt(np.nanmean((abs_diff_10m/stereo_depths_10m), axis=(1, 2))**2)) * 100  # RMSPE formula for values within 10m
    rmse_20p = (np.sqrt(np.nanmean((abs_diff_20m/stereo_depths_20m), axis=(1, 2))**2)) * 100  # RMSPE formula for values within 20m
    rmse_40p = (np.sqrt(np.nanmean((abs_diff_40m/stereo_depths_40m), axis=(1, 2))**2)) * 100  # RMSPE formula for values within 40m

    return rmse, rmse_10m, rmse_20m, rmse_40m, rmse_p, rmse_10p, rmse_20p, rmse_40p



