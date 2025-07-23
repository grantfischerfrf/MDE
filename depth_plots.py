import sys
sys.path.append('../')
from pointcloud.main import pull_data, comp_timestamps, find_ref_dict

import cv2
import os
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import mpl_scatter_density
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable


#TODO: consider making a plot utils script, turn the masking into functions
#TODO: change the names of data from sp1, sp2 to something actually readable
#TODO: save the data after clipping, then load. Save the computational time.


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
                horizon_candidates.append(line[0])

    # select top most line and extend it to the borders of the image
    if horizon_candidates:
        top_line = min(horizon_candidates, key=lambda line: min(line[1], line[3]))
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


def four_panel_plot(sp1, sp2, sp3, sp4, img_path:str, output_path:str, title: str):

    # Creates a four panel plot
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(sp1, aspect='auto')
    ax1.set_title('Raw Image', fontsize=22)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 1])
    plot2 = ax2.imshow(sp2, cmap='tab20', alpha=0.75, aspect='auto', vmin=0, vmax=80)
    ax2.imshow(sp1, alpha=0.35, aspect='auto')
    ax2.set_title('Instantaneous Depth', fontsize=22)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[1, 0])
    plot3 = ax3.imshow(sp3, cmap='coolwarm', alpha=0.85, aspect='auto', vmin=-1, vmax=1)
    ax3.imshow(sp1, alpha=0.35, aspect='auto')
    ax3.set_title('Instantaneous Velocity', fontsize=22)
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = fig.add_subplot(gs[1, 1])
    plot4 = ax4.imshow(sp4, cmap='OrRd', norm=colors.LogNorm(), aspect='auto')
    ax4.set_title('2* Standard Deviation', fontsize=22)
    ax4.set_xticks([])
    ax4.set_yticks([])

    cbar = fig.colorbar(plot2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04, ticks=np.arange(0, 84, 4)[::2])
    cbar.ax.tick_params(labelsize=15)
    cbar2 = fig.colorbar(plot3, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=15)
    cbar3 = fig.colorbar(plot4, ax=ax4, orientation='vertical', fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=15)
    cbar.set_label('Depth (m)', fontsize=15)
    cbar2.set_label('Velocity (m/s)', fontsize=15)
    cbar3.set_label('2 * Standard Deviation (m/s)', fontsize=15)

    fig.suptitle(title, fontsize=24, fontweight='bold')

    plt.tight_layout()
    # plt.savefig(output_path + os.path.basename(img_path).split('.')[0] + '_velocity_map.png', dpi=200)
    plt.savefig(output_path + img_path + '.png', dpi=200)
    plt.close('all')


def four_panel_gcp_velocity(sp1, sp2, sp3, sp4, UV, ind, UV_vel, img_path, output_path):

    # plot velocity maps
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(sp1, aspect='auto')
    ax1.set_title('Raw Image', fontsize=22)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 1])
    plot2 = ax2.imshow(sp2, cmap='tab20', alpha=0.75, aspect='auto', vmin=0, vmax=80)
    ax2.imshow(sp1, alpha=0.35, aspect='auto')
    ax2.set_title('Instantaneous Depth (rollingAvg)', fontsize=22)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[1, 0])
    plot3 = ax3.imshow(sp3, cmap='coolwarm', alpha=0.85, aspect='auto', vmin=-1, vmax=1)
    ax3.imshow(sp1, alpha=0.35, aspect='auto')
    ax3.plot(UV[0], UV[1], 'r+', markersize=8)
    for i in range(len(UV[0])):
        ax3.annotate(f'{ind[i]}: {UV_vel[i]:.2f} m/s', xy=(UV[0][i], UV[1][i]), color='white', fontsize=10, fontweight='bold')
    ax3.set_title('Instantaneous Velocity (rollingAvg)', fontsize=22)
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = fig.add_subplot(gs[1, 1])
    plot4 = ax4.imshow(sp4, cmap='OrRd', norm=colors.LogNorm(), aspect='auto')
    ax4.set_title('2* Standard Deviation (m/s)', fontsize=22)
    ax4.set_xticks([])
    ax4.set_yticks([])

    cbar = fig.colorbar(plot2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04, ticks=np.arange(0, 84, 4)[::2])
    cbar.ax.tick_params(labelsize=15)
    cbar2 = fig.colorbar(plot3, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=15)
    cbar3 = fig.colorbar(plot4, ax=ax4, orientation='vertical', fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=15)
    cbar.set_label('Depth (m)', fontsize=15)
    cbar2.set_label('Velocity (m/s)', fontsize=15)
    cbar3.set_label('2 * Standard Deviation (m/s)', fontsize=15)

    plt.tight_layout()
    plt.savefig(output_path + os.path.basename(img_path).split('.')[0] + '_gcp_velocity_map.png', dpi=200)
    plt.close('all')


def error_plot(estimated_depth, calculated_depth, rmse, date, camera, output_path, vmax=60):

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot([calculated_depth[i][1] for i in range(len(calculated_depth))], estimated_depth, 'ro', markersize=5)
    for i in range(len(calculated_depth)):
        ax.annotate(f'{int(calculated_depth[i][0])}', xy=(calculated_depth[i][1], estimated_depth[i]), xytext=(4, -1), textcoords='offset points', color='black', fontsize=5.5)
    ax.set_xlabel('Calculated GCP Depth (m)')
    ax.set_ylabel('Estimated Depth (m)')
    ax.set_title(f'{str(date.year)}{date.month:02d}{date.day:02d}{camera} Est v Cal GCP Depths: RMSE = {rmse:.2f}m')
    ax.plot([0, vmax], [0, vmax], 'k--')  # line y=x for reference
    plt.savefig(f'{os.path.dirname(output_path.rstrip('/'))}/outputs/gcp/{str(date.year)}{date.month:02d}{date.day:02d}_{camera}' + '_gcp_error.png', dpi = 180)
    # plt.show()
    plt.close('all')


def error_comparison(est_depths:list, cal_depths:list, labels:list, rmse:list, output_path, vmax=80):

    #est_depths, cal_depths should be a list of lists
    #labels should be a list of strings

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    for i in range(len(labels)):
        ax.scatter(cal_depths[i], est_depths[i], s=20, marker='o', label=labels[i])

    ax.set_xlabel('Calculated GCP Depth (m)')
    ax.set_ylabel('Estimated Depth (m)')
    ax.set_title('Estimated vs Calculated GCP Depths')
    ax.plot([0, vmax], [0, vmax], 'k--')  # line y=x for reference
    ax.text(0.05, 0.95, f'Bob RMSE: {rmse[0]:.2f}m\nMary RMSE: {rmse[1]:.2f}m', transform=ax.transAxes, fontsize=12,verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend()
    plt.savefig(f'{os.path.dirname(output_path.rstrip('/'))}/outputs/gcp/allData' + '_gcp_depth_comparison.png', dpi = 180)
    plt.close('all')


def panel_plot(data_folder:str, output_path:str):

    # load_data
    RGB = True
    RGB_dict_list = load_data(cam='RGB', dataset='surf', data_folder=data_folder)
    # rename dictionary keys
    for d in RGB_dict_list:
        d['name'] = d['name'] + '_rgb'

    NIR_dict_list = load_data(cam='NIR', dataset='surf', data_folder=data_folder)

    dicts = comp_timestamps(RGB_dict_list + NIR_dict_list)

    # background images
    rgb_image = dicts['aux_image_rect_color_rgb']['messages']  # RGB image
    nir_image = dicts['left_image_rect']['messages']  # NIR image

    # mde predictions
    rgb_mde_depth = dicts['aux_image_rect_color_mde_rgb']['messages']  # RGB MDE Depth
    nir_mde_depth = dicts['left_image_rect_mde']['messages']  # NIR MDE depth

    # stereo depth maps
    rgb_stereo_depth = dicts['left_depth_rgb']['messages']  # RGB StereoDepth
    nir_stereo_depth = dicts['left_depth']['messages']  # NIR StereoDepth

    # create masks above horizon, pull horizon line
    rgb_masks = []
    rgb_horizons = []
    nir_masks = []
    nir_horizons = []
    for i in range(len(rgb_image)):
        rgb_mask, rgb_horizon = mask_horizon(rgb_image[i])
        rgb_masks.append(rgb_mask)
        rgb_horizons.append(rgb_horizon)
        nir_mask, nir_horizon = mask_horizon(nir_image[i])
        nir_masks.append(nir_mask)
        nir_horizons.append(nir_horizon)
    rgb_masks = np.array(rgb_masks)
    rgb_horizons = np.array(rgb_horizons)
    nir_masks = np.array(nir_masks)
    nir_horizons = np.array(nir_horizons)

    # MDE prediction is run off of RGB image. Stereo is calculated from monochrome cams. Offset in camera location, must shift to correct for it.
    rgb_stereo_depth = shift_data(rgb_stereo_depth, num_pixels=16)  # Shift stereo depth map to match MDE depth map

    # create copies of depth maps for computation and masking
    rgb_mde_depth_ = rgb_mde_depth.copy()
    rgb_stereo_depth_ = rgb_stereo_depth.copy()
    nir_mde_depth_ = nir_mde_depth.copy()
    nir_stereo_depth_ = nir_stereo_depth.copy()

    # apply mask to depth maps and background image
    if rgb_masks.ndim > 3:
        rgb_masks = rgb_masks[:, :, :, 0]  # Remove the channel dimension if it exists
    for i in range(len(rgb_mde_depth)):
        rgb_mde_depth_[i] = np.where(rgb_masks[i] != 255, rgb_mde_depth_[i], np.nan)
        nir_mde_depth_[i] = np.where(nir_masks[i] != 255, nir_mde_depth_[i], np.nan)
        rgb_stereo_depth_[i] = np.where(rgb_masks[i] != 255, rgb_stereo_depth_[i], np.nan)
        nir_stereo_depth_[i] = np.where(nir_masks[i] != 255, nir_stereo_depth_[i], np.nan)

    # create box around the LARC, turn into NaNs
    yy, xx = np.meshgrid(np.arange(rgb_mde_depth_.shape[1]), np.arange(rgb_mde_depth_.shape[2]),
                         indexing='ij')  # RGB and NIR images should be the same size

    rgb_larc_mask = ((xx >= 500) & (yy >= 530)) | (xx >= 800)  # creates 600x960 mask
    rgb_larc_mask = np.broadcast_to(rgb_larc_mask,
                                    rgb_mde_depth_.shape)  # broadcast 2d larc_mask to shape of all depth maps nx600x960
    rgb_mde_depth_[rgb_larc_mask] = np.nan  # apply mask to MDE depth map
    rgb_stereo_depth_[rgb_larc_mask] = np.nan  # apply mask to Stereo depth map

    nir_larc_mask = ((xx >= 370) & (yy >= 400)) | ((620 <= xx) & (xx <= 720) & (yy >= 120))
    nir_larc_mask = np.broadcast_to(nir_larc_mask,
                                    nir_mde_depth_.shape)  # broadcast 2d larc_mask to shape of all depth maps
    nir_mde_depth_[nir_larc_mask] = np.nan  # apply mask to MDE depth map
    nir_stereo_depth_[nir_larc_mask] = np.nan  # apply mask to Stereo depth map

    # mask out values greater than or equal to 80m
    rgb_mde_depth_[rgb_mde_depth_ >= 80] = np.nan
    rgb_stereo_depth_[rgb_stereo_depth_ >= 80] = np.nan
    nir_mde_depth_[nir_mde_depth_ >= 80] = np.nan
    nir_stereo_depth_[nir_stereo_depth_ >= 80] = np.nan

    # bin stereo depths per meter
    num_bins = np.arange(0, 81, 1)
    rgb_stereo_depth_binned = np.digitize(rgb_stereo_depth_.flatten(), num_bins).reshape(rgb_stereo_depth_.shape)
    nir_stereo_depth_binned = np.digitize(nir_stereo_depth_.flatten(), num_bins).reshape(nir_stereo_depth_.shape)
    # calculate mean and std dev of MDE predictions in each bin
    rgb_mde_mean = []
    rgb_mde_std_dev = []
    nir_mde_mean = []
    nir_mde_std_dev = []
    rgb_X = []
    nir_X = []

    print('Processing...')
    for i in tqdm(range(len(num_bins))):
        rgb_mask = rgb_stereo_depth_binned == i
        nir_mask = nir_stereo_depth_binned == i
        if np.any(rgb_mask):
            rgb_mask = np.where(rgb_stereo_depth_binned == i, rgb_mde_depth_, np.nan)
            rgb_mde_mean.append(np.nanmean(rgb_mask, axis=(1, 2)))
            rgb_mde_std_dev.append(np.nanstd(rgb_mask, axis=(1, 2)))
            rgb_X.append(num_bins[i])
        if np.any(nir_mask):
            nir_mask = np.where(nir_stereo_depth_binned == i, nir_mde_depth_, np.nan)
            nir_mde_mean.append(np.nanmean(nir_mask, axis=(1, 2)))
            nir_mde_std_dev.append(np.nanstd(nir_mask, axis=(1, 2)))
            nir_X.append(num_bins[i])

    # difference stereo and mde to plot
    rgb_diff = rgb_mde_depth_ - rgb_stereo_depth_
    nir_diff = nir_mde_depth_ - nir_stereo_depth_

    combined_min = np.nanmin(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
    combined_max = np.nanmax(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
    bins = np.linspace(combined_min, combined_max, 205)  # define edges for consistent bin widths

    for i in range(len(rgb_mde_depth)):
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 3)

        #mde map
        ax1 = fig.add_subplot(gs[0, 0])
        plot1 = ax1.imshow(rgb_mde_depth[i], cmap='viridis', alpha=0.75, aspect='equal', vmin=0, vmax=80)
        ax1.imshow(rgb_image[i], alpha=0.35, aspect='equal')
        ax1.plot([rgb_horizons[i][0], rgb_horizons[i][1]], [rgb_horizons[i][2], rgb_horizons[i][3]], color='red',linewidth=3)
        ax1.set_title('RGB MDE Depth Map', fontsize=22)
        ax1.set_xticks([])
        ax1.set_yticks([])

        #rgb stereo map
        ax1b = fig.add_subplot(gs[0, 1])
        plot1b = ax1b.imshow(rgb_stereo_depth[i], cmap='viridis', alpha=0.75, aspect='auto', vmin=0, vmax=80)
        ax1b.imshow(rgb_image[i], alpha=0.35, aspect='equal')
        ax1b.plot([rgb_horizons[i][0], rgb_horizons[i][1]], [rgb_horizons[i][2], rgb_horizons[i][3]], color='red',linewidth=3)
        ax1b.set_title('RGB Stereo Depth Map', fontsize=22)
        ax1b.set_xticks([])
        ax1b.set_yticks([])

        #NIR MDE map
        ax2 = fig.add_subplot(gs[1, 0])
        plot2 = ax2.imshow(nir_mde_depth[i], cmap='viridis', alpha=0.75, aspect='auto', vmin=0, vmax=80)
        ax2.imshow(nir_image[i], alpha=0.35, aspect='equal')
        ax2.plot([nir_horizons[i][0], nir_horizons[i][1]], [nir_horizons[i][2], nir_horizons[i][3]], color='red',linewidth=3)
        ax2.set_title('NIR MDE Depth Map', fontsize=22)
        ax2.set_xticks([])
        ax2.set_yticks([])

        #NIR stereo map
        ax2b = fig.add_subplot(gs[1, 1])
        plot2b = ax2b.imshow(nir_stereo_depth[i], cmap='viridis', alpha=0.75, aspect='auto', vmin=0, vmax=80)
        ax2b.imshow(nir_image[i], alpha=0.35, aspect='equal')
        ax2b.plot([nir_horizons[i][0], nir_horizons[i][1]], [nir_horizons[i][2], nir_horizons[i][3]], color='red',linewidth=3)
        ax2b.set_title('NIR Stereo Depth Map', fontsize=22)
        ax2b.set_xticks([])
        ax2b.set_yticks([])

        # fill overlay
        ax3 = fig.add_subplot(gs[0, 2])

        rgb_mde_mean_i = np.array(rgb_mde_mean)[:, i]
        rgb_mde_std_dev_i = np.array(rgb_mde_std_dev)[:, i]
        nir_mde_mean_i = np.array(nir_mde_mean)[:, i]
        nir_mde_std_dev_i = np.array(nir_mde_std_dev)[:, i]

        ax3.plot(rgb_X, rgb_mde_mean_i + rgb_mde_std_dev_i, color='red', linestyle='--', linewidth=2,label='+ 1 Standard Deviation')
        ax3.plot(rgb_X, rgb_mde_mean_i - rgb_mde_std_dev_i, color='blue', linestyle='--', linewidth=2,label='- 1 Standard Deviation')
        ax3.plot(rgb_X, rgb_mde_mean_i + 2 * rgb_mde_std_dev_i, color='red', linewidth=2, label='+ 2 Standard Deviation')
        ax3.plot(rgb_X, rgb_mde_mean_i - 2 * rgb_mde_std_dev_i, color='blue', linewidth=2,label='- 2 Standard Deviation')
        ax3.plot(rgb_X, rgb_mde_mean_i, color='black', linewidth=2, label='MDE Mean Depth')
        ax3.fill_between(rgb_X, rgb_mde_mean_i + 2 * rgb_mde_std_dev_i, rgb_mde_mean_i - 2 * rgb_mde_std_dev_i, color='orange', alpha=0.5, label='RGB')

        ax3.plot(nir_X, nir_mde_mean_i + nir_mde_std_dev_i, color='red', linestyle='--', linewidth=2)
        ax3.plot(nir_X, nir_mde_mean_i - nir_mde_std_dev_i, color='blue', linestyle='--', linewidth=2)
        ax3.plot(nir_X, nir_mde_mean_i + 2 * nir_mde_std_dev_i, color='red', linewidth=2)
        ax3.plot(nir_X, nir_mde_mean_i - 2 * nir_mde_std_dev_i, color='blue', linewidth=2)
        ax3.plot(nir_X, nir_mde_mean_i, color='black', linewidth=2)
        ax3.fill_between(nir_X, nir_mde_mean_i + 2 * nir_mde_std_dev_i, nir_mde_mean_i - 2 * nir_mde_std_dev_i, color='cyan', alpha=0.5, label='NIR')

        ax3.set_xlabel('Stereo Depth (m)', fontsize=22)
        ax3.set_ylabel('MDE Depth (m)', fontsize=22)
        ax3.set_title('MDE v Stereo Depth Estimation', fontsize=25)
        ax3.set_xlim(0, 80)
        ax3.set_ylim(0, 80)
        ax3.plot([0, 80], [0, 80], 'k--')  # line y=x for reference
        ax3.tick_params(axis='both', which='major', labelsize=18)
        ax3.legend(loc = 'upper left', fontsize=9)

        # stacked histogram
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(rgb_diff[i].flatten(), bins=bins, color='orange', edgecolor='black', density=True, alpha=0.5,label='RGB')
        ax4.hist(nir_diff[i].flatten(), bins=bins, color='cyan', edgecolor='black', density=True, alpha=0.5, label='NIR')

        ax4.set_xlabel('Error (m)', fontsize=20)
        ax4.set_ylabel('Probability Density', fontsize=20)
        ax4.set_title('MDE - Stereo Depth Error Distribution', fontsize=22)
        ax4.set_xlim(-20, 20)
        ax4.set_ylim(0, 0.4)
        ax4.tick_params(axis='both', which='major', labelsize=15)
        ax4.legend(fontsize=18)

        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig.colorbar(plot1, cax=cax1, orientation='vertical', fraction=0.046, pad=0.04)  # , ticks=np.arange(0, 84, 4)[::2])
        cbar1.ax.tick_params(labelsize=15)
        cbar1.set_label('Estimated Depth (m)', fontsize=15)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = fig.colorbar(plot2, cax=cax2, orientation='vertical', fraction=0.046, pad=0.04)  # , ticks=np.arange(0, 84, 4)[::2])
        cbar2.ax.tick_params(labelsize=15)
        cbar2.set_label('Estimated Depth (m)', fontsize=15)

        fig.suptitle(f'Error Metrics', fontsize=22, fontweight='bold', y=1.001)

        #calculate difference in timestamps between RGB and NIR image
        rgb_timestamp = dicts['aux_image_rect_color_rgb']['timestamps'][i]
        nir_timestamp = dicts['left_image_rect']['timestamps'][i]
        time_diff = abs(rgb_timestamp - nir_timestamp)

        plt.tight_layout()
        plt.savefig(output_path + str(time_diff) + '.png', dpi=180)
        plt.close('all')


def panel_plot_allData(data_folder:str, output_path:str):

    # load_data
    RGB = True
    RGB_dict_list = load_data(cam='RGB', dataset=None, data_folder=data_folder)
    # rename dictionary keys
    for d in RGB_dict_list:
        d['name'] = d['name'] + '_rgb'

    NIR_dict_list = load_data(cam='NIR', dataset=None, data_folder=data_folder)

    dicts = comp_timestamps(RGB_dict_list + NIR_dict_list)

    # background images
    rgb_image = dicts['aux_image_rect_color_rgb']['messages']  # RGB image
    nir_image = dicts['left_image_rect']['messages']  # NIR image

    # mde predictions
    rgb_mde_depth = dicts['aux_image_rect_color_mde_rgb']['messages']  # RGB MDE Depth
    nir_mde_depth = dicts['left_image_rect_mde']['messages']  # NIR MDE depth

    # stereo depth maps
    rgb_stereo_depth = dicts['left_depth_rgb']['messages']  # RGB StereoDepth
    nir_stereo_depth = dicts['left_depth']['messages']  # NIR StereoDepth

    # create masks above horizon, pull horizon line
    rgb_masks = []
    rgb_horizons = []
    nir_masks = []
    nir_horizons = []
    for i in range(len(rgb_image)):
        rgb_mask, rgb_horizon = mask_horizon(rgb_image[i])
        rgb_masks.append(rgb_mask)
        rgb_horizons.append(rgb_horizon)
        nir_mask, nir_horizon = mask_horizon(nir_image[i])
        nir_masks.append(nir_mask)
        nir_horizons.append(nir_horizon)
    rgb_masks = np.array(rgb_masks)
    rgb_horizons = np.array(rgb_horizons)
    nir_masks = np.array(nir_masks)
    nir_horizons = np.array(nir_horizons)

    # MDE prediction is run off of RGB image. Stereo is calculated from monochrome cams. Offset in camera location, must shift to correct for it.
    rgb_stereo_depth = shift_data(rgb_stereo_depth, num_pixels=16)  # Shift stereo depth map to match MDE depth map

    # create copies of depth maps for computation and masking
    rgb_mde_depth_ = rgb_mde_depth.copy()
    rgb_stereo_depth_ = rgb_stereo_depth.copy()
    nir_mde_depth_ = nir_mde_depth.copy()
    nir_stereo_depth_ = nir_stereo_depth.copy()

    # apply mask to depth maps and background image
    if rgb_masks.ndim > 3:
        rgb_masks = rgb_masks[:, :, :, 0]  # Remove the channel dimension if it exists
    for i in range(len(rgb_mde_depth)):
        rgb_mde_depth_[i] = np.where(rgb_masks[i] != 255, rgb_mde_depth_[i], np.nan)
        nir_mde_depth_[i] = np.where(nir_masks[i] != 255, nir_mde_depth_[i], np.nan)
        rgb_stereo_depth_[i] = np.where(rgb_masks[i] != 255, rgb_stereo_depth_[i], np.nan)
        nir_stereo_depth_[i] = np.where(nir_masks[i] != 255, nir_stereo_depth_[i], np.nan)

    # create box around the LARC, turn into NaNs
    yy, xx = np.meshgrid(np.arange(rgb_mde_depth_.shape[1]), np.arange(rgb_mde_depth_.shape[2]),
                         indexing='ij')  # RGB and NIR images should be the same size

    rgb_larc_mask = ((xx >= 500) & (yy >= 530)) | (xx >= 800)  # creates 600x960 mask
    rgb_larc_mask = np.broadcast_to(rgb_larc_mask,
                                    rgb_mde_depth_.shape)  # broadcast 2d larc_mask to shape of all depth maps nx600x960
    rgb_mde_depth_[rgb_larc_mask] = np.nan  # apply mask to MDE depth map
    rgb_stereo_depth_[rgb_larc_mask] = np.nan  # apply mask to Stereo depth map

    nir_larc_mask = ((xx >= 370) & (yy >= 400)) | ((620 <= xx) & (xx <= 720) & (yy >= 120))
    nir_larc_mask = np.broadcast_to(nir_larc_mask,
                                    nir_mde_depth_.shape)  # broadcast 2d larc_mask to shape of all depth maps
    nir_mde_depth_[nir_larc_mask] = np.nan  # apply mask to MDE depth map
    nir_stereo_depth_[nir_larc_mask] = np.nan  # apply mask to Stereo depth map

    # mask out values greater than or equal to 80m
    rgb_mde_depth_[rgb_mde_depth_ >= 80] = np.nan
    rgb_stereo_depth_[rgb_stereo_depth_ >= 80] = np.nan
    nir_mde_depth_[nir_mde_depth_ >= 80] = np.nan
    nir_stereo_depth_[nir_stereo_depth_ >= 80] = np.nan

    # bin stereo depths per meter
    num_bins = np.arange(0, 81, 1)
    rgb_stereo_depth_binned = np.digitize(rgb_stereo_depth_.flatten(), num_bins).reshape(rgb_stereo_depth_.shape)
    nir_stereo_depth_binned = np.digitize(nir_stereo_depth_.flatten(), num_bins).reshape(nir_stereo_depth_.shape)
    # calculate mean and std dev of MDE predictions in each bin
    rgb_mde_mean = []
    rgb_mde_std_dev = []
    nir_mde_mean = []
    nir_mde_std_dev = []
    rgb_X = []
    nir_X = []

    print('Processing...')
    for i in tqdm(range(len(num_bins))):
        rgb_mask = rgb_stereo_depth_binned == i
        nir_mask = nir_stereo_depth_binned == i
        if np.any(rgb_mask):
            rgb_mask = np.where(rgb_stereo_depth_binned == i, rgb_mde_depth_, np.nan)
            rgb_mde_mean.append(np.nanmean(rgb_mask, axis=(1, 2)))
            rgb_mde_std_dev.append(np.nanstd(rgb_mask, axis=(1, 2)))
            rgb_X.append(num_bins[i])
        if np.any(nir_mask):
            nir_mask = np.where(nir_stereo_depth_binned == i, nir_mde_depth_, np.nan)
            nir_mde_mean.append(np.nanmean(nir_mask, axis=(1, 2)))
            nir_mde_std_dev.append(np.nanstd(nir_mask, axis=(1, 2)))
            nir_X.append(num_bins[i])

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2)

    # rgb mde map
    ax1 = fig.add_subplot(gs[0, 0])
    plot1 = ax1.imshow(rgb_mde_depth[-1], cmap='viridis', alpha=0.75, aspect='auto', vmin=0, vmax=80)
    ax1.imshow(rgb_image[-1], alpha=0.35, aspect='equal')
    ax1.plot([rgb_horizons[-1][0], rgb_horizons[-1][1]], [rgb_horizons[-1][2], rgb_horizons[-1][3]], color='red', linewidth=3)
    ax1.set_title('RGB MDE Depth Map', fontsize=22)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # nir mde map
    ax2 = fig.add_subplot(gs[0, 1])
    plot2 = ax2.imshow(nir_mde_depth[-1], cmap='viridis', alpha=0.75, aspect='auto', vmin=0, vmax=80)
    ax2.imshow(nir_image[-1], alpha=0.35, aspect='equal')
    ax2.plot([nir_horizons[-1][0], nir_horizons[-1][1]], [nir_horizons[-1][2], nir_horizons[-1][3]], color='red', linewidth=3)
    ax2.set_title('NIR MDE Depth Map', fontsize=22)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # fill overlay
    ax3 = fig.add_subplot(gs[1, 0])
    rgb_mde_mean_i = np.nanmean(np.array(rgb_mde_mean), axis=1)
    rgb_mde_std_dev_i = np.nanmean(np.array(rgb_mde_std_dev), axis=1)
    nir_mde_mean_i = np.nanmean(np.array(nir_mde_mean), axis=1)
    nir_mde_std_dev_i = np.nanmean(np.array(nir_mde_std_dev), axis=1)

    ax3.plot(rgb_X, rgb_mde_mean_i + rgb_mde_std_dev_i, color='red', linestyle='--', linewidth=2,label='+ 1 Standard Deviation')
    ax3.plot(rgb_X, rgb_mde_mean_i - rgb_mde_std_dev_i, color='blue', linestyle='--', linewidth=2,label='- 1 Standard Deviation')
    ax3.plot(rgb_X, rgb_mde_mean_i + 2 * rgb_mde_std_dev_i, color='red', linewidth=2, label='+ 2 Standard Deviation')
    ax3.plot(rgb_X, rgb_mde_mean_i - 2 * rgb_mde_std_dev_i, color='blue', linewidth=2, label='- 2 Standard Deviation')
    ax3.plot(rgb_X, rgb_mde_mean_i, color='black', linewidth=2, label='MDE Mean Depth')
    ax3.fill_between(rgb_X, rgb_mde_mean_i + 2 * rgb_mde_std_dev_i, rgb_mde_mean_i - 2 * rgb_mde_std_dev_i, color='orange', alpha=0.5, label='RGB')
    ax3.plot(nir_X, nir_mde_mean_i + nir_mde_std_dev_i, color='red', linestyle='--', linewidth=2)
    ax3.plot(nir_X, nir_mde_mean_i - nir_mde_std_dev_i, color='blue', linestyle='--', linewidth=2)
    ax3.plot(nir_X, nir_mde_mean_i + 2 * nir_mde_std_dev_i, color='red', linewidth=2)
    ax3.plot(nir_X, nir_mde_mean_i - 2 * nir_mde_std_dev_i, color='blue', linewidth=2)
    ax3.plot(nir_X, nir_mde_mean_i, color='black', linewidth=2)
    ax3.fill_between(nir_X, nir_mde_mean_i + 2 * nir_mde_std_dev_i, nir_mde_mean_i - 2 * nir_mde_std_dev_i, color='cyan',alpha=0.5, label='NIR')

    ax3.set_xlabel('Stereo Depth (m)', fontsize=22)
    ax3.set_ylabel('MDE Depth (m)', fontsize=22)
    ax3.set_title('MDE v Stereo Mean Depth Estimation', fontsize=25)
    ax3.set_xlim(0, 80)
    ax3.set_ylim(0, 80)
    ax3.plot([0, 80], [0, 80], 'k--')  # line y=x for reference
    ax3.tick_params(axis='both', which='major', labelsize=18)
    ax3.legend(loc='upper left', fontsize=12)

    #stacked histogram
    ax4 = fig.add_subplot(gs[1, 1])
    rgb_diff = rgb_mde_depth_ - rgb_stereo_depth_
    nir_diff = nir_mde_depth_ - nir_stereo_depth_
    rgb_diff_mean = np.nanmean(rgb_diff, axis=(1, 2))
    nir_diff_mean = np.nanmean(nir_diff, axis=(1, 2))

    combined_min = np.nanmin(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
    combined_max = np.nanmax(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
    bins = np.linspace(combined_min, combined_max, 69)  # define edges for consistent bin widths

    ax4.hist(rgb_diff_mean, bins=bins, color='orange', edgecolor='black', density=True, alpha=0.5, label='RGB')
    ax4.hist(nir_diff_mean, bins=bins, color='cyan', edgecolor='black', density=True, alpha=0.5, label='NIR')

    ax4.set_xlabel('Error (m)', fontsize=20)
    ax4.set_ylabel('Probability Density', fontsize=20)
    ax4.set_title('MDE - Stereo Mean Depth Error Distribution', fontsize=22)
    ax4.set_xlim(-15, 15)
    ax4.set_ylim(0, 0.3)
    ax4.tick_params(axis='both', which='major', labelsize=15)
    ax4.legend(fontsize=18)

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(plot1, cax=cax1, orientation='vertical', fraction=0.046, pad=0.04)  # , ticks=np.arange(0, 84, 4)[::2])
    cbar1.ax.tick_params(labelsize=15)
    cbar1.set_label('Estimated Depth (m)', fontsize=15)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(plot2, cax=cax2, orientation='vertical', fraction=0.046, pad=0.04)  # , ticks=np.arange(0, 84, 4)[::2])
    cbar2.ax.tick_params(labelsize=15)
    cbar2.set_label('Estimated Depth (m)', fontsize=15)

    fig.suptitle(f'Error Metrics', fontsize=22, fontweight='bold', y=1.000)

    plt.tight_layout()
    plt.savefig(output_path + 'DepthComp_allData_openwater' + '.png', dpi=180)
    plt.close('all')


def depth_comp(data_folder:str, output_path:str):

    #pull data
    RGB = True
    dict_list = load_data(cam='NIR', dataset=None, data_folder=data_folder)


    dicts = comp_timestamps(dict_list)

    #pull background images
    # sp3 = dicts['aux_image_rect_color']['messages']  # RGB Image
    sp3 = dicts['left_image_rect']['messages']  # NIR Image

    #pull depth maps
    sp1 = dicts['left_image_rect_mde']['messages']  # NIR MDE Depth
    # sp1 = dicts['aux_image_rect_color_mde']['messages']  # RGB MDE Depth
    sp2 = dicts['left_depth']['messages']  # StereoDepth

    #create masks above horizon, pull horizon line
    masks = []
    horizons = []
    for i in range(len(sp3)):
        mask, horizon = mask_horizon(sp3[i])
        masks.append(mask)
        horizons.append(horizon)
    masks = np.array(masks)
    horizons = np.array(horizons)

    # MDE prediction is run off of RGB image. Stereo is calculated from monochrome cams. Offset in camera location, must shift to correct for it.
    if RGB:
        sp2 = shift_data(sp2, num_pixels=16)  # Shift stereo depth map to match MDE depth map

    #create copies of depth maps for computation and masking
    sp1_ = sp1.copy()
    sp2_ = sp2.copy()

    #apply mask to depth maps and background image
    if masks.ndim > 3:
        masks = masks[:, :, :, 0]  # Remove the channel dimension if it exists
    for i in range(len(sp1)):
        sp1_[i] = np.where(masks[i] != 255, sp1_[i], np.nan)
        sp2_[i] = np.where(masks[i] != 255, sp2_[i], np.nan)

    # #create box around the LARC, turn into NaNs
    if RGB:
        yy, xx = np.meshgrid(np.arange(sp1_.shape[1]), np.arange(sp1_.shape[2]), indexing='ij')
        larc_mask = ((xx >= 500) & (yy >= 530)) | (xx >= 800) #creates 600x960 mask
        larc_mask = np.broadcast_to(larc_mask, sp1_.shape)  #broadcast 2d larc_mask to shape of all depth maps
        sp1_[larc_mask] = np.nan #apply mask to MDE depth map
        sp2_[larc_mask] = np.nan #apply mask to Stereo depth map

    if not RGB:
        yy, xx = np.meshgrid(np.arange(sp1_.shape[1]), np.arange(sp1_.shape[2]), indexing='ij')
        larc_mask = ((xx >= 370) & (yy >= 400)) | ((620 <= xx) & (xx <= 720) & (yy >=120))
        larc_mask = np.broadcast_to(larc_mask, sp1_.shape)  # broadcast 2d larc_mask to shape of all depth maps
        sp1_[larc_mask] = np.nan  # apply mask to MDE depth map
        sp2_[larc_mask] = np.nan  # apply mask to Stereo depth map

    #mask out values greater than or equal to 80m
    sp1_[sp1_ >= 80] = np.nan
    sp2_[sp2_ >= 80] = np.nan

    sp4 = np.abs(sp2_ - sp1_) # Absolute Difference between MDE and StereoDepth

    #calculate rmse of difference map
    rmse = np.sqrt(np.nanmean((sp4**2), axis=(1, 2))) # Calculate RMSE for each depth map
    #calculate different RMSE values at different depths
    sp4_10m = np.where(sp4 < 10, sp4, np.nan)  # Mask values greater than or equal to 10m
    rmse_10m = np.sqrt(np.nanmean((sp4_10m**2), axis=(1, 2)))  # Calculate RMSE for each depth map within 10m
    sp4_20m = np.where(sp4 < 20, sp4, np.nan)  # Mask values greater than or equal to 20m
    rmse_20m = np.sqrt(np.nanmean((sp4_20m**2), axis=(1, 2)))  # Calculate RMSE for each depth map within 20m
    sp4_40m = np.where(sp4 < 40, sp4, np.nan)  # Mask values greater than or equal to 40m
    rmse_40m = np.sqrt(np.nanmean((sp4_40m**2), axis=(1, 2)))  # Calculate RMSE for each depth map within 40m

    #RMSPE - average magnitude of error as a percentage of the acutal values
    sp2_10m = np.where(sp2_ < 10, sp2_, np.nan)  # Mask values greater than or equal to 10m
    sp2_20m = np.where(sp2_ < 20, sp2_, np.nan)  # Mask values greater than or equal to 20m
    sp2_40m = np.where(sp2_ < 40, sp2_, np.nan)  # Mask values greater than or equal to 40m
    rmse_p = (np.sqrt(np.nanmean(((sp2_ - sp1_)/sp2_), axis=(1, 2))**2)) * 100  #RMSPE formula
    rmse_10p = (np.sqrt(np.nanmean((sp4_10m/sp2_10m), axis=(1, 2))**2)) * 100  # RMSPE formula for values within 10m
    rmse_20p = (np.sqrt(np.nanmean((sp4_20m/sp2_20m), axis=(1, 2))**2)) * 100  # RMSPE formula for values within 20m
    rmse_40p = (np.sqrt(np.nanmean((sp4_40m/sp2_40m), axis=(1, 2))**2)) * 100  # RMSPE formula for values within 40m


    #plot
    for i in range(len(sp1)):
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        plot1 = ax1.imshow(sp1[i], cmap='viridis', alpha=0.75, aspect='equal', vmin=0, vmax=80)
        ax1.imshow(sp3[i], alpha=0.35, aspect='equal')
        ax1.plot([horizons[i][0], horizons[i][1]], [horizons[i][2], horizons[i][3]], color='red', linewidth=3)
        ax1.set_title('NIR MDE Depth Map', fontsize=22)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[0, 1])
        plot2 = ax2.imshow(sp2[i], cmap = 'viridis', alpha=0.75, aspect='equal', vmin=0, vmax=80)
        ax2.imshow(sp3[i], alpha=0.35, aspect='equal')
        ax2.plot([horizons[i][0], horizons[i][1]], [horizons[i][2], horizons[i][3]], color='red', linewidth=3)
        ax2.text(0.05, 0.95,
                 f'RMSE within 10m: {rmse_10m[i]:.2f} m\nRMSE within 20m: {rmse_20m[i]:.2f} m\nRMSE within 40m: {rmse_40m[i]:.2f} m\nRMSE within 80m: {rmse[i]:.2f} m',
                 transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        ax2.text(0.45, 0.95,
                 f'RMSPE within 10m: {rmse_10p[i]:.2f} %\nRMSPE within 20m: {rmse_20p[i]:.2f} %\nRMSPE within 40m: {rmse_40p[i]:.2f} %\nRMSPE within 80m: {rmse_p[i]:.2f} %',
                 transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        ax2.set_title('Stereo Depth Map', fontsize=22)
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = fig.add_subplot(gs[1, 0])
        plot3 = ax3.imshow(sp4[i], cmap='Greys', aspect='equal', vmin=0, vmax=10)
        ax3.set_title(f'Difference (abs value)', fontsize=22)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ins_ax3 = inset_axes(ax3, width='25%', height='25%', loc='upper left', bbox_to_anchor=(0.08, 0.23, .75, .75), bbox_transform=ax3.transAxes)
        ins_ax3.plot([10, 20, 40, 80], [rmse_10m[i], rmse_20m[i], rmse_40m[i], rmse[i]], marker='o', color='blue', linewidth=2, markersize=5, label='RMSE')
        ins_ax3.set_xticks([10, 20, 40, 80])
        ins_ax3.set_ylim(0,10)
        ins_ax3.tick_params(axis='x', labelrotation=45, labelsize=8)
        ins_ax3.set_xlabel('Depth(m)')
        ins_ax3.set_ylabel('RMSE (m)')
        #create second y axis with RMSPE values
        ins_ax3_2 = ins_ax3.twinx()
        ins_ax3_2.plot([10, 20, 40, 80], [rmse_10p[i], rmse_20p[i], rmse_40p[i], rmse_p[i]], marker='o', color='orange', linewidth=2, markersize=5, label='RMSPE')
        ins_ax3_2.set_ylim(0, 15)
        ins_ax3_2.set_yticks([0, 5, 10, 15])
        ins_ax3_2.set_ylabel('RMSPE (%)')
        plots = ins_ax3.get_lines() + ins_ax3_2.get_lines()
        labels = [p.get_label() for p in plots]
        ins_ax3.legend(plots, labels, loc='upper right', fontsize=8, bbox_to_anchor=(2.3, 1.0))


        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(sp3[i], aspect='equal')
        ax4.plot([horizons[i][0], horizons[i][1]], [horizons[i][2], horizons[i][3]], color='red', linewidth=3)
        # if RGB:
        #     ax4.plot([500, 500], [599, 530], color='red', linewidth=3)
        #     ax4.plot([500, 800], [530, 530], color='red', linewidth=3)
        #     ax4.plot([800, 800], [530, 0], color='red', linewidth=3)
        # if not RGB:
        #     ax4.plot([370, 370], [599, 400], color='red', linewidth=3)
        #     ax4.plot([370, 620], [400, 400], color='red', linewidth=3)
        #     ax4.plot([620, 620], [400, 120], color='red', linewidth=3)
        #     ax4.plot([620, 720], [120, 120], color='red', linewidth=3)
        #     ax4.plot([720, 720], [400, 120], color='red', linewidth=3)
        #     ax4.plot([720, 959], [400, 400], color='red', linewidth=3)
        ax4.set_title('NIR Image', fontsize=22)
        ax4.set_xticks([])
        ax4.set_yticks([])

        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig.colorbar(plot1, cax=cax1, orientation='vertical', fraction=0.046, pad=0.04)#, ticks=np.arange(0, 84, 4)[::2])
        cbar1.ax.tick_params(labelsize=15)
        cbar1.set_label('Estimated Depth (m)', fontsize=15)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = fig.colorbar(plot2, cax=cax2, orientation='vertical', fraction=0.046, pad=0.04)#, ticks=np.arange(0, 84, 4)[::2])
        cbar2.ax.tick_params(labelsize=15)
        cbar2.set_label('Estimated Depth (m)', fontsize=15)

        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cbar3 = fig.colorbar(plot3, cax=cax3, orientation='vertical', fraction=0.046, pad=0.04, ticks=np.arange(0, 11, 1))
        cbar3.ax.tick_params(labelsize=15)
        cbar3.set_label('Depth Difference (m)', fontsize=15)

        fig.suptitle('Depth Comparison: NIR MDE vs Stereo Depth', fontsize=24, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
        plt.close('all')


def one_to_one(data_folder:str, output_path:str):

    #pull data - rgb
    RGB = True
    dict_list = load_data(cam='RGB', dataset='surf', data_folder=data_folder)

    dicts = comp_timestamps(dict_list)

    mde_depth = dicts['aux_image_rect_color_mde']['messages']  # MDE Depth
    # mde_depth = dicts['left_image_rect_mde']['messages']  # MDE Depth
    stereo_depth = dicts['left_depth']['messages']  # StereoDepth
    raw_images = dicts['aux_image_rect_color']['messages']  # RGB Image
    # raw_images = dicts['left_image_rect']['messages']  # NIR Image


    # create masks above horizon, pull horizon line
    masks = []
    horizons = []
    for i in range(len(raw_images)):
        mask, horizon = mask_horizon(raw_images[i])
        masks.append(mask)
        horizons.append(horizon)
    masks = np.array(masks)


    # MDE prediction is run off of RGB image. Stereo is calculated from monochrome cams. Offset in camera location, must shift to correct for it.
    if RGB:
        stereo_depth = shift_data(stereo_depth, num_pixels=16)  # Shift stereo depth map to match MDE depth map

    # apply mask to depth maps and background image
    if masks.ndim > 3:
        masks = masks[:, :, :, 0]  # Remove the channel dimension if it exists
    for i in range(len(mde_depth)):
        mde_depth[i] = np.where(masks[i] != 255, mde_depth[i], np.nan)
        stereo_depth[i] = np.where(masks[i] != 255, stereo_depth[i], np.nan)

    # #create box around the LARC, turn into NaNs
    if RGB:
        yy, xx = np.meshgrid(np.arange(mde_depth.shape[1]), np.arange(mde_depth.shape[2]), indexing='ij')
        larc_mask = ((xx >= 500) & (yy >= 530)) | (xx >= 800)  # creates 600x960 mask
        larc_mask = np.broadcast_to(larc_mask, mde_depth.shape)  # broadcast 2d larc_mask to shape of all depth maps
        mde_depth[larc_mask] = np.nan  # apply mask to MDE depth map
        stereo_depth[larc_mask] = np.nan  # apply mask to Stereo depth map

    if not RGB:
        yy, xx = np.meshgrid(np.arange(mde_depth.shape[1]), np.arange(mde_depth.shape[2]), indexing='ij')
        larc_mask = ((xx >= 370) & (yy >= 400)) | ((620 <= xx) & (xx <= 720) & (yy >= 120))
        larc_mask = np.broadcast_to(larc_mask, mde_depth.shape)  # broadcast 2d larc_mask to shape of all depth maps
        mde_depth[larc_mask] = np.nan  # apply mask to MDE depth map
        stereo_depth[larc_mask] = np.nan  # apply mask to Stereo depth map

    #mask out values greater than or equal to 80m
    mde_depth[mde_depth >= 80] = np.nan
    stereo_depth[stereo_depth >= 80] = np.nan

    rmse = np.sqrt(np.nanmean((stereo_depth - mde_depth) ** 2, axis=(1, 2)))  # Calculate RMSE for each depth map

    # #plot one to one plot of stereo and mde depth maps
    # for i in range(len(mde_depth)):
    #     fig = plt.figure(figsize=(20, 20))
    #     ax = fig.add_subplot(111)
    #
    #     ax.plot(stereo_depth[i], mde_depth[i], 'ro', markersize=5, alpha=0.1)
    #     ax.set_xlabel('Stereo Depth (m)', fontsize=15)
    #     ax.set_ylabel('MDE Depth (m)', fontsize=15)
    #     ax.set_title(f'Stereo vs MDE Depth: RMSE = {rmse[i]:.2f}m', fontsize=22)
    #     ax.plot([0, 80], [0, 80], 'k--')  # line y=x for reference
    #     ax.set_xlim(0, 80)
    #     ax.set_ylim(0, 80)
    #     plt.savefig(output_path + str(dicts['aux_image_rect_color']['timestamps'][i]) + '.png', dpi=180)
    #     plt.tight_layout()
    #     plt.close('all')

    # plot scatter plot of all points
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    for i in range(len(mde_depth)):
        ax.plot(stereo_depth[i][::10, ::10], mde_depth[i][::10, ::10], 'r+', markersize=3, alpha=0.1) #scatter plot
    ax.set_xlabel('Stereo Depth (m)', fontsize=15)
    ax.set_ylabel('MDE Depth (m)', fontsize=15)
    ax.set_title(f'Stereo vs MDE Depth: RMSE = {np.mean(rmse):.2f}m', fontsize=22)
    ax.plot([0, 80], [0, 80], 'k--')  # line y=x for reference
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    plt.savefig(output_path + 'RGB_all_points.png', dpi=180)
    plt.close('all')


def histogram(data_folder:str, output_path:str):

    #pull data - rgb
    RGB = True
    dict_list = load_data(cam='RGB', dataset='surf', data_folder=data_folder)

    dicts = comp_timestamps(dict_list)

    #pull background images
    sp3 = dicts['aux_image_rect_color']['messages']  # RGB Image
    # sp3 = dicts['left_image_rect']['messages']  # NIR Image

    #pull depth maps
    # sp1 = dicts['left_image_rect_mde']['messages']  # NIR MDE Depth
    sp1 = dicts['aux_image_rect_color_mde']['messages']  # RGB MDE Depth
    sp2 = dicts['left_depth']['messages']  # StereoDepth

    #create masks above horizon, pull horizon line
    masks = []
    horizons = []
    for i in range(len(sp3)):
        mask, horizon = mask_horizon(sp3[i])
        masks.append(mask)
        horizons.append(horizon)
    masks = np.array(masks)

    # MDE prediction is run off of RGB image. Stereo is calculated from monochrome cams. Offset in camera location, must shift to correct for it.
    if RGB:
        sp2 = shift_data(sp2, num_pixels=16)  # Shift stereo depth map to match MDE depth map

    #create copies of depth maps for computation and masking
    sp1_ = sp1.copy()
    sp2_ = sp2.copy()

    #apply mask to depth maps and background image
    if masks.ndim > 3:
        masks = masks[:, :, :, 0]  # Remove the channel dimension if it exists
    for i in range(len(sp1)):
        sp1_[i] = np.where(masks[i] != 255, sp1_[i], np.nan)
        sp2_[i] = np.where(masks[i] != 255, sp2_[i], np.nan)

    # #create box around the LARC, turn into NaNs
    if RGB:
        yy, xx = np.meshgrid(np.arange(sp1_.shape[1]), np.arange(sp1_.shape[2]), indexing='ij')
        larc_mask = ((xx >= 500) & (yy >= 530)) | (xx >= 800) #creates 600x960 mask
        larc_mask = np.broadcast_to(larc_mask, sp1_.shape)  #broadcast 2d larc_mask to shape of all depth maps
        sp1_[larc_mask] = np.nan #apply mask to MDE depth map
        sp2_[larc_mask] = np.nan #apply mask to Stereo depth map

    if not RGB:
        yy, xx = np.meshgrid(np.arange(sp1_.shape[1]), np.arange(sp1_.shape[2]), indexing='ij')
        larc_mask = ((xx >= 370) & (yy >= 400)) | ((620 <= xx) & (xx <= 720) & (yy >=120))
        larc_mask = np.broadcast_to(larc_mask, sp1_.shape)  # broadcast 2d larc_mask to shape of all depth maps
        sp1_[larc_mask] = np.nan  # apply mask to MDE depth map
        sp2_[larc_mask] = np.nan  # apply mask to Stereo depth map

    #mask out values greater than or equal to 80m
    sp1_[sp1_ >= 80] = np.nan
    sp2_[sp2_ >= 80] = np.nan

    sp4 = np.abs(sp2_ - sp1_) # Absolute Difference between MDE and StereoDepth

    for i in range(len(sp1_)):
        #create histogram of all points
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        ax.hist(sp1_[i].flatten() - sp2_[i].flatten(), bins=68, color='blue', edgecolor='black', density=True)
        ax.set_xlabel('Error (m)', fontsize=20)
        ax.set_ylabel('Probability Density', fontsize=20)
        ax.set_title('RGB MDE v Stereo Depth Error', fontsize=22)
        ax.set_xlim(-40, 40)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.savefig(output_path + str(dicts['aux_image_rect_color']['timestamps'][i]) + '.png', dpi=180)
        plt.close('all')


def heatmap(data_folder:str, output_path:str):

    # create a box and whisker plot based on standard deviation at each point on the x axis
    # pull data - rgb
    RGB = False
    dict_list = load_data(cam='RGB', dataset='surf', data_folder=data_folder)

    dicts = comp_timestamps(dict_list)

    # pull background images
    sp3 = dicts['aux_image_rect_color']['messages']  # RGB Image
    # sp3 = dicts['left_image_rect']['messages']  # NIR Image

    # pull depth maps
    # sp1 = dicts['left_image_rect_mde']['messages']  # NIR MDE Depth
    sp1 = dicts['aux_image_rect_color_mde']['messages']  # RGB MDE Depth
    sp2 = dicts['left_depth']['messages']  # StereoDepth

    # create masks above horizon, pull horizon line
    masks = []
    horizons = []
    for i in range(len(sp3)):
        mask, horizon = mask_horizon(sp3[i])
        masks.append(mask)
        horizons.append(horizon)
    masks = np.array(masks)

    # MDE prediction is run off of RGB image. Stereo is calculated from monochrome cams. Offset in camera location, must shift to correct for it.
    if RGB:
        sp2 = shift_data(sp2, num_pixels=16)  # Shift stereo depth map to match MDE depth map

    # create copies of depth maps for computation and masking
    sp1_ = sp1.copy()
    sp2_ = sp2.copy()

    # apply mask to depth maps and background image
    if masks.ndim > 3:
        masks = masks[:, :, :, 0]  # Remove the channel dimension if it exists
    for i in range(len(sp1)):
        sp1_[i] = np.where(masks[i] != 255, sp1_[i], np.nan)
        sp2_[i] = np.where(masks[i] != 255, sp2_[i], np.nan)

    # #create box around the LARC, turn into NaNs
    if RGB:
        yy, xx = np.meshgrid(np.arange(sp1_.shape[1]), np.arange(sp1_.shape[2]), indexing='ij')
        larc_mask = ((xx >= 500) & (yy >= 530)) | (xx >= 800)  # creates 600x960 mask
        larc_mask = np.broadcast_to(larc_mask, sp1_.shape)  # broadcast 2d larc_mask to shape of all depth maps
        sp1_[larc_mask] = np.nan  # apply mask to MDE depth map
        sp2_[larc_mask] = np.nan  # apply mask to Stereo depth map

    if not RGB:
        yy, xx = np.meshgrid(np.arange(sp1_.shape[1]), np.arange(sp1_.shape[2]), indexing='ij')
        larc_mask = ((xx >= 370) & (yy >= 400)) | ((620 <= xx) & (xx <= 720) & (yy >= 120))
        larc_mask = np.broadcast_to(larc_mask, sp1_.shape)  # broadcast 2d larc_mask to shape of all depth maps
        sp1_[larc_mask] = np.nan  # apply mask to MDE depth map
        sp2_[larc_mask] = np.nan  # apply mask to Stereo depth map

    # mask out values greater than or equal to 80m
    sp1_[sp1_ >= 80] = np.nan
    sp2_[sp2_ >= 80] = np.nan
    #
    # # plot scatter density of all points
    # for i in range(len(sp1_)):
    #
    #     #filter out nans
    #     mde_depth = sp1_[i][(~np.isnan(sp1_[i])) & (~np.isnan(sp2_[i]))]
    #     stereo_depth = sp2_[i][(~np.isnan(sp1_[i])) & (~np.isnan(sp2_[i]))]
    #
    #     fig = plt.figure(figsize=(20, 20))
    #     ax = fig.add_subplot(111)
    #     ax.hist2d(stereo_depth.flatten(), mde_depth.flatten(), bins=68, cmap='hot_r', norm=colors.LogNorm(), range=((0, 80), (0, 80)))
    #     ax.set_xlabel('Stereo Depth (m)', fontsize=20)
    #     ax.set_ylabel('MDE Depth (m)', fontsize=20)
    #     ax.set_title('Stereo v MDE depth Heatmap', fontsize=22)
    #     ax.set_xlim(0, 80)
    #     ax.set_ylim(0, 80)
    #     ax.plot([0, 80], [0, 80], color='magenta', linestyle='--')  # line y=x for reference
    #     ax.tick_params(axis='both', which='major', labelsize=15)
    #
    #     #colorbar
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     cbar = fig.colorbar(ax.collections[0], cax=cax, orientation='vertical', fraction=0.046, pad=0.04)
    #     cbar.ax.tick_params(labelsize=15)
    #     cbar.set_label('Count', fontsize=20)
    #
    #     plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
    #     plt.close('all')

    #create heatmap for total dataset
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)

    mde_depth = sp1_[(~np.isnan(sp1_)) & (~np.isnan(sp2_))]
    stereo_depth = sp2_[(~np.isnan(sp1_)) & (~np.isnan(sp2_))]

    ax.hist2d(stereo_depth.flatten(), mde_depth.flatten(), bins=136, cmap='hot_r', norm=colors.LogNorm(), range=((0, 80), (0, 80)))
    ax.set_xlabel('Stereo Depth (m)', fontsize=20)
    ax.set_ylabel('MDE Depth (m)', fontsize=20)
    ax.set_title('Stereo v MDE depth Heatmap', fontsize=22)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.plot([0, 80], [0, 80], color='magenta', linestyle='--')  # line y=x for reference
    ax.tick_params(axis='both', which='major', labelsize=15)

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(ax.collections[0], cax=cax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Count', fontsize=20)

    plt.savefig(output_path + 'total_heatmap.png', dpi=180)
    plt.close('all')


def fill_plot(data_folder:str, output_path:str):

    #create a box and whisker plot based on standard deviation at each point on the x axis
    #pull data - rgb
    RGB = False
    dict_list = load_data(cam='NIR', dataset=None, data_folder=data_folder)

    dicts = comp_timestamps(dict_list)

    #pull background images
    # sp3 = dicts['aux_image_rect_color']['messages']  # RGB Image
    sp3 = dicts['left_image_rect']['messages']  # NIR Image

    #pull depth maps
    sp1 = dicts['left_image_rect_mde']['messages']  # NIR MDE Depth
    # sp1 = dicts['aux_image_rect_color_mde']['messages']  # RGB MDE Depth
    sp2 = dicts['left_depth']['messages']  # StereoDepth

    #create masks above horizon, pull horizon line
    masks = []
    horizons = []
    for i in range(len(sp3)):
        mask, horizon = mask_horizon(sp3[i])
        masks.append(mask)
        horizons.append(horizon)
    masks = np.array(masks)

    # MDE prediction is run off of RGB image. Stereo is calculated from monochrome cams. Offset in camera location, must shift to correct for it.
    if RGB:
        sp2 = shift_data(sp2, num_pixels=16)  # Shift stereo depth map to match MDE depth map

    #create copies of depth maps for computation and masking
    sp1_ = sp1.copy()
    sp2_ = sp2.copy()

    #apply mask to depth maps and background image
    if masks.ndim > 3:
        masks = masks[:, :, :, 0]  # Remove the channel dimension if it exists
    for i in range(len(sp1)):
        sp1_[i] = np.where(masks[i] != 255, sp1_[i], np.nan)
        sp2_[i] = np.where(masks[i] != 255, sp2_[i], np.nan)

    # #create box around the LARC, turn into NaNs
    if RGB:
        yy, xx = np.meshgrid(np.arange(sp1_.shape[1]), np.arange(sp1_.shape[2]), indexing='ij')
        larc_mask = ((xx >= 500) & (yy >= 530)) | (xx >= 800) #creates 600x960 mask
        larc_mask = np.broadcast_to(larc_mask, sp1_.shape)  #broadcast 2d larc_mask to shape of all depth maps nx600x960
        sp1_[larc_mask] = np.nan #apply mask to MDE depth map
        sp2_[larc_mask] = np.nan #apply mask to Stereo depth map

    if not RGB:
        yy, xx = np.meshgrid(np.arange(sp1_.shape[1]), np.arange(sp1_.shape[2]), indexing='ij')
        larc_mask = ((xx >= 370) & (yy >= 400)) | ((620 <= xx) & (xx <= 720) & (yy >=120))
        larc_mask = np.broadcast_to(larc_mask, sp1_.shape)  # broadcast 2d larc_mask to shape of all depth maps
        sp1_[larc_mask] = np.nan  # apply mask to MDE depth map
        sp2_[larc_mask] = np.nan  # apply mask to Stereo depth map

    #mask out values greater than or equal to 80m
    sp1_[sp1_ >= 80] = np.nan
    sp2_[sp2_ >= 80] = np.nan

    sp4 = np.abs(sp2_ - sp1_) # Absolute Difference between MDE and StereoDepth

    #bin stereo depths per meter
    num_bins = np.arange(0, 81, 1)
    sp2_binned = np.digitize(sp2_.flatten(), num_bins).reshape(sp2_.shape)
    #calculate mean and std dev of MDE predictions in each bin
    mde_mean = []
    mde_std_dev = []
    X = []

    print('Processing...')
    for i in tqdm(range(len(num_bins))):
        mask = sp2_binned == i
        if np.any(mask):
            mask = np.where(sp2_binned == i, sp1_, np.nan)
            mde_mean.append(np.nanmean(mask, axis=(1, 2)))
            mde_std_dev.append(np.nanstd(mask, axis=(1, 2)))
            X.append(num_bins[i])

    print('Plotting...')
    # create fill between plot
    for i in range(np.array(mde_mean).shape[1]):
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)

        mde_mean_i = np.array(mde_mean)[:, i]
        mde_std_dev_i = np.array(mde_std_dev)[:, i]

        ax.plot(X, mde_mean_i + mde_std_dev_i, color='red', linestyle='--', linewidth=2, label='+ 1 Standard Deviation')
        ax.plot(X, mde_mean_i - mde_std_dev_i, color='blue', linestyle='--', linewidth=2, label='- 1 Standard Deviation')
        ax.plot(X, mde_mean_i + 2*mde_std_dev_i, color='red', linewidth=2, label='+ 2 Standard Deviation')
        ax.plot(X, mde_mean_i - 2*mde_std_dev_i, color='blue', linewidth=2, label='- 2 Standard Deviation')
        ax.plot(X, mde_mean_i, color='black', linewidth=2, label='MDE Mean Depth')
        ax.fill_between(X, mde_mean_i + 2*mde_std_dev_i, mde_mean_i - 2*mde_std_dev_i, color='green', alpha=0.2)
        ax.set_xlabel('Stereo Depth (m)', fontsize=22)
        ax.set_ylabel('MDE Depth (m)', fontsize=22)
        ax.set_title('Stereo v MDE Depth Mean and Standard Deviation', fontsize=25)
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 80)
        ax.plot([0, 80], [0, 80], 'k--')  # line y=x for reference
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.legend(fontsize=20)

        plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
        plt.close('all')


def fill_overlay(data_folder:str, output_path:str):

    # load_data
    RGB = True
    RGB_dict_list = load_data(cam='RGB', dataset='surf', data_folder=data_folder)
    #rename dictionary keys
    for d in RGB_dict_list:
        d['name'] = d['name'] + '_rgb'

    NIR_dict_list = load_data(cam='NIR', dataset='surf', data_folder=data_folder)

    dicts = comp_timestamps(RGB_dict_list + NIR_dict_list)

    #background images
    rgb_image = dicts['aux_image_rect_color_rgb']['messages']  # RGB image
    nir_image = dicts['left_image_rect']['messages'] # NIR image

    #mde predictions
    rgb_mde_depth = dicts['aux_image_rect_color_mde_rgb']['messages'] # RGB MDE Depth
    nir_mde_depth = dicts['left_image_rect_mde']['messages']  # NIR MDE depth

    #stereo depth maps
    rgb_stereo_depth = dicts['left_depth_rgb']['messages']  # RGB StereoDepth
    nir_stereo_depth = dicts['left_depth']['messages']  # NIR StereoDepth

    #create masks above horizon, pull horizon line
    rgb_masks = []
    nir_masks = []
    for i in range(len(rgb_image)):
        rgb_mask, _ = mask_horizon(rgb_image[i])
        rgb_masks.append(rgb_mask)
        nir_mask, _ = mask_horizon(nir_image[i])
        nir_masks.append(nir_mask)
    rgb_masks = np.array(rgb_masks)
    nir_masks = np.array(nir_masks)

    # MDE prediction is run off of RGB image. Stereo is calculated from monochrome cams. Offset in camera location, must shift to correct for it.
    rgb_stereo_depth = shift_data(rgb_stereo_depth, num_pixels=16)  # Shift stereo depth map to match MDE depth map

    # create copies of depth maps for computation and masking
    rgb_mde_depth_ = rgb_mde_depth.copy()
    rgb_stereo_depth_ = rgb_stereo_depth.copy()
    nir_mde_depth_ = nir_mde_depth.copy()
    nir_stereo_depth_ = nir_stereo_depth.copy()

    # apply mask to depth maps and background image
    if rgb_masks.ndim > 3:
        rgb_masks = rgb_masks[:, :, :, 0]  # Remove the channel dimension if it exists
    for i in range(len(rgb_mde_depth)):
        rgb_mde_depth_[i] = np.where(rgb_masks[i] != 255, rgb_mde_depth_[i], np.nan)
        nir_mde_depth_[i] = np.where(nir_masks[i] != 255, nir_mde_depth_[i], np.nan)
        rgb_stereo_depth[i] = np.where(rgb_masks[i] != 255, rgb_stereo_depth[i], np.nan)
        nir_stereo_depth[i] = np.where(nir_masks[i] != 255, nir_stereo_depth[i], np.nan)

    #create box around the LARC, turn into NaNs
    yy, xx = np.meshgrid(np.arange(rgb_mde_depth_.shape[1]), np.arange(rgb_mde_depth_.shape[2]), indexing='ij') # RGB and NIR images should be the same size

    rgb_larc_mask = ((xx >= 500) & (yy >= 530)) | (xx >= 800)  # creates 600x960 mask
    rgb_larc_mask = np.broadcast_to(rgb_larc_mask, rgb_mde_depth_.shape)  # broadcast 2d larc_mask to shape of all depth maps nx600x960
    rgb_mde_depth_[rgb_larc_mask] = np.nan  # apply mask to MDE depth map
    rgb_stereo_depth_[rgb_larc_mask] = np.nan  # apply mask to Stereo depth map

    nir_larc_mask = ((xx >= 370) & (yy >= 400)) | ((620 <= xx) & (xx <= 720) & (yy >= 120))
    nir_larc_mask = np.broadcast_to(nir_larc_mask, nir_mde_depth_.shape)  # broadcast 2d larc_mask to shape of all depth maps
    nir_mde_depth_[nir_larc_mask] = np.nan  # apply mask to MDE depth map
    nir_stereo_depth_[nir_larc_mask] = np.nan  # apply mask to Stereo depth map

    # mask out values greater than or equal to 80m
    rgb_mde_depth_[rgb_mde_depth_ >= 80] = np.nan
    rgb_stereo_depth_[rgb_stereo_depth_ >= 80] = np.nan
    nir_mde_depth_[nir_mde_depth_ >= 80] = np.nan
    nir_stereo_depth_[nir_stereo_depth_ >= 80] = np.nan

    # bin stereo depths per meter
    num_bins = np.arange(0, 81, 1)
    rgb_stereo_depth_binned = np.digitize(rgb_stereo_depth_.flatten(), num_bins).reshape(rgb_stereo_depth_.shape)
    nir_stereo_depth_binned = np.digitize(nir_stereo_depth_.flatten(), num_bins).reshape(nir_stereo_depth_.shape)
    # calculate mean and std dev of MDE predictions in each bin
    rgb_mde_mean = []
    rgb_mde_std_dev = []
    nir_mde_mean = []
    nir_mde_std_dev = []
    rgb_X = []
    nir_X = []

    print('Processing...')
    for i in tqdm(range(len(num_bins))):
        rgb_mask = rgb_stereo_depth_binned == i
        nir_mask = nir_stereo_depth_binned ==i
        if np.any(rgb_mask):
            rgb_mask = np.where(rgb_stereo_depth_binned == i, rgb_mde_depth_, np.nan)
            rgb_mde_mean.append(np.nanmean(rgb_mask, axis=(1, 2)))
            rgb_mde_std_dev.append(np.nanstd(rgb_mask, axis=(1, 2)))
            rgb_X.append(num_bins[i])
        if np.any(nir_mask):
            nir_mask = np.where(nir_stereo_depth_binned == i, nir_mde_depth_, np.nan)
            nir_mde_mean.append(np.nanmean(nir_mask, axis=(1, 2)))
            nir_mde_std_dev.append(np.nanstd(nir_mask, axis=(1, 2)))
            nir_X.append(num_bins[i])

    print('Plotting...')
    # # create fill between plot
    # for i in range(np.array(rgb_mde_mean).shape[1]):
    #     fig = plt.figure(figsize=(20, 20))
    #     ax = fig.add_subplot(111)
    #
    #     rgb_mde_mean_i = np.array(rgb_mde_mean)[:, i]
    #     rgb_mde_std_dev_i = np.array(rgb_mde_std_dev)[:, i]
    #     nir_mde_mean_i = np.array(nir_mde_mean)[:, i]
    #     nir_mde_std_dev_i = np.array(nir_mde_std_dev)[:, i]
    #
    #     ax.plot(rgb_X, rgb_mde_mean_i + rgb_mde_std_dev_i, color='red', linestyle='--', linewidth=2, label='+ 1 Standard Deviation')
    #     ax.plot(rgb_X, rgb_mde_mean_i - rgb_mde_std_dev_i, color='blue', linestyle='--', linewidth=2,
    #             label='- 1 Standard Deviation')
    #     ax.plot(rgb_X, rgb_mde_mean_i + 2 * rgb_mde_std_dev_i, color='red', linewidth=2, label='+ 2 Standard Deviation')
    #     ax.plot(rgb_X, rgb_mde_mean_i - 2 * rgb_mde_std_dev_i, color='blue', linewidth=2, label='- 2 Standard Deviation')
    #     ax.plot(rgb_X, rgb_mde_mean_i, color='black', linewidth=2, label='MDE Mean Depth')
    #     ax.fill_between(rgb_X, rgb_mde_mean_i + 2 * rgb_mde_std_dev_i, rgb_mde_mean_i - 2 * rgb_mde_std_dev_i, color='orange', alpha=0.5, label = 'RGB')
    #
    #     ax.plot(nir_X, nir_mde_mean_i + nir_mde_std_dev_i, color='red', linestyle='--', linewidth=2)
    #     ax.plot(nir_X, nir_mde_mean_i - nir_mde_std_dev_i, color='blue', linestyle='--', linewidth=2)
    #     ax.plot(nir_X, nir_mde_mean_i + 2 * nir_mde_std_dev_i, color='red', linewidth=2)
    #     ax.plot(nir_X, nir_mde_mean_i - 2 * nir_mde_std_dev_i, color='blue', linewidth=2)
    #     ax.plot(nir_X, nir_mde_mean_i, color='black', linewidth=2)
    #     ax.fill_between(nir_X, nir_mde_mean_i + 2 * nir_mde_std_dev_i, nir_mde_mean_i - 2 * nir_mde_std_dev_i, color='cyan', alpha=0.5, label = 'NIR')
    #
    #     ax.set_xlabel('Stereo Depth (m)', fontsize=22)
    #     ax.set_ylabel('MDE Depth (m)', fontsize=22)
    #     ax.set_title('Stereo v MDE Depth Mean and Standard Deviation', fontsize=25)
    #     ax.set_xlim(0, 80)
    #     ax.set_ylim(0, 80)
    #     ax.plot([0, 80], [0, 80], 'k--')  # line y=x for reference
    #     ax.tick_params(axis='both', which='major', labelsize=18)
    #     ax.legend(fontsize=20)
    #
    #     plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
    #     plt.close('all')


    #plot fill overlay for the entire dataset
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    rgb_mde_mean_i = np.nanmean(np.array(rgb_mde_mean), axis=1)
    rgb_mde_std_dev_i = np.nanmean(np.array(rgb_mde_std_dev), axis=1)
    nir_mde_mean_i = np.nanmean(np.array(nir_mde_mean), axis=1)
    nir_mde_std_dev_i = np.nanmean(np.array(nir_mde_std_dev), axis=1)

    ax.plot(rgb_X, rgb_mde_mean_i + rgb_mde_std_dev_i, color='red', linestyle='--', linewidth=2, label='+ 1 Standard Deviation')
    ax.plot(rgb_X, rgb_mde_mean_i - rgb_mde_std_dev_i, color='blue', linestyle='--', linewidth=2, label='- 1 Standard Deviation')
    ax.plot(rgb_X, rgb_mde_mean_i + 2 * rgb_mde_std_dev_i, color='red', linewidth=2, label='+ 2 Standard Deviation')
    ax.plot(rgb_X, rgb_mde_mean_i - 2 * rgb_mde_std_dev_i, color='blue', linewidth=2, label='- 2 Standard Deviation')
    ax.plot(rgb_X, rgb_mde_mean_i, color='black', linewidth=2, label='MDE Mean Depth')
    ax.fill_between(rgb_X, rgb_mde_mean_i + 2 * rgb_mde_std_dev_i, rgb_mde_mean_i - 2 * rgb_mde_std_dev_i, color='orange', alpha=0.5, label = 'RGB')
    ax.plot(nir_X, nir_mde_mean_i + nir_mde_std_dev_i, color='red', linestyle='--', linewidth=2)
    ax.plot(nir_X, nir_mde_mean_i - nir_mde_std_dev_i, color='blue', linestyle='--', linewidth=2)
    ax.plot(nir_X, nir_mde_mean_i + 2 * nir_mde_std_dev_i, color='red', linewidth=2)
    ax.plot(nir_X, nir_mde_mean_i - 2 * nir_mde_std_dev_i, color='blue', linewidth=2)
    ax.plot(nir_X, nir_mde_mean_i, color='black', linewidth=2)
    ax.fill_between(nir_X, nir_mde_mean_i + 2 * nir_mde_std_dev_i, nir_mde_mean_i - 2 * nir_mde_std_dev_i, color='cyan', alpha=0.5, label = 'NIR')

    ax.set_xlabel('Stereo Depth (m)', fontsize=22)
    ax.set_ylabel('MDE Depth (m)', fontsize=22)
    ax.set_title('Stereo v MDE Depth Mean and Standard Deviation', fontsize=25)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.plot([0, 80], [0, 80], 'k--')  # line y=x for reference
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=20)

    plt.savefig(output_path + 'fill_overlay_allPoints' + '.png', dpi=180)
    plt.close('all')


def one_to_one_overlay(data_folder:str, output_path:str):

    #overlay RGB and NIR for one to one plots
    RGB_dict_list = load_data(cam='RGB', dataset='surf', data_folder=data_folder)
    # rename dictionary keys
    for d in RGB_dict_list:
        d['name'] = d['name'] + '_rgb'

    NIR_dict_list = load_data(cam='NIR', dataset='surf', data_folder=data_folder)

    dicts = comp_timestamps(RGB_dict_list + NIR_dict_list)

    #background images
    rgb_image = dicts['aux_image_rect_color_rgb']['messages']  # RGB image
    nir_image = dicts['left_image_rect']['messages'] # NIR image

    #mde predictions
    rgb_mde_depth = dicts['aux_image_rect_color_mde_rgb']['messages'] # RGB MDE Depth
    nir_mde_depth = dicts['left_image_rect_mde']['messages']  # NIR MDE depth

    #stereo depth maps
    rgb_stereo_depth = dicts['left_depth_rgb']['messages']  # RGB StereoDepth
    nir_stereo_depth = dicts['left_depth']['messages']  # NIR StereoDepth

    #create masks above horizon, pull horizon line
    rgb_masks = []
    nir_masks = []
    for i in range(len(rgb_image)):
        rgb_mask, _ = mask_horizon(rgb_image[i])
        rgb_masks.append(rgb_mask)
        nir_mask, _ = mask_horizon(nir_image[i])
        nir_masks.append(nir_mask)
    rgb_masks = np.array(rgb_masks)
    nir_masks = np.array(nir_masks)

    # MDE prediction is run off of RGB image. Stereo is calculated from monochrome cams. Offset in camera location, must shift to correct for it.
    rgb_stereo_depth = shift_data(rgb_stereo_depth, num_pixels=16)  # Shift stereo depth map to match MDE depth map

    # create copies of depth maps for computation and masking
    rgb_mde_depth_ = rgb_mde_depth.copy()
    rgb_stereo_depth_ = rgb_stereo_depth.copy()
    nir_mde_depth_ = nir_mde_depth.copy()
    nir_stereo_depth_ = nir_stereo_depth.copy()

    # apply mask to depth maps and background image
    if rgb_masks.ndim > 3:
        rgb_masks = rgb_masks[:, :, :, 0]  # Remove the channel dimension if it exists
    for i in range(len(rgb_mde_depth)):
        rgb_mde_depth_[i] = np.where(rgb_masks[i] != 255, rgb_mde_depth_[i], np.nan)
        nir_mde_depth_[i] = np.where(nir_masks[i] != 255, nir_mde_depth_[i], np.nan)
        rgb_stereo_depth[i] = np.where(rgb_masks[i] != 255, rgb_stereo_depth[i], np.nan)
        nir_stereo_depth[i] = np.where(nir_masks[i] != 255, nir_stereo_depth[i], np.nan)

    #create box around the LARC, turn into NaNs
    yy, xx = np.meshgrid(np.arange(rgb_mde_depth_.shape[1]), np.arange(rgb_mde_depth_.shape[2]), indexing='ij') # RGB and NIR images should be the same size

    rgb_larc_mask = ((xx >= 500) & (yy >= 530)) | (xx >= 800)  # creates 600x960 mask
    rgb_larc_mask = np.broadcast_to(rgb_larc_mask, rgb_mde_depth_.shape)  # broadcast 2d larc_mask to shape of all depth maps nx600x960
    rgb_mde_depth_[rgb_larc_mask] = np.nan  # apply mask to MDE depth map
    rgb_stereo_depth_[rgb_larc_mask] = np.nan  # apply mask to Stereo depth map

    nir_larc_mask = ((xx >= 370) & (yy >= 400)) | ((620 <= xx) & (xx <= 720) & (yy >= 120))
    nir_larc_mask = np.broadcast_to(nir_larc_mask, nir_mde_depth_.shape)  # broadcast 2d larc_mask to shape of all depth maps
    nir_mde_depth_[nir_larc_mask] = np.nan  # apply mask to MDE depth map
    nir_stereo_depth_[nir_larc_mask] = np.nan  # apply mask to Stereo depth map

    # mask out values greater than or equal to 80m
    rgb_mde_depth_[rgb_mde_depth_ >= 80] = np.nan
    rgb_stereo_depth_[rgb_stereo_depth_ >= 80] = np.nan
    nir_mde_depth_[nir_mde_depth_ >= 80] = np.nan
    nir_stereo_depth_[nir_stereo_depth_ >= 80] = np.nan

    rgb_rmse = np.sqrt(np.nanmean((rgb_stereo_depth_ - rgb_mde_depth_)**2, axis=(1, 2)))
    nir_rmse = np.sqrt(np.nanmean((nir_stereo_depth_ - nir_mde_depth_)**2, axis=(1, 2)))

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    print('plotting...')
    for i in range(len(rgb_mde_depth_)):
        ax.plot(rgb_stereo_depth_[i][::20, ::20], rgb_mde_depth_[i][::20, ::20],'r+', markersize=3, alpha=0.1, label='RGB')
        ax.plot(nir_stereo_depth_[i][::20, ::20], nir_mde_depth_[i][::20, ::20],'b+', markersize=3, alpha=0.1, label='NIR')

    ax.annotate(f'RGB RMSE: {np.nanmean(rgb_rmse):.2f} m', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=20, color='red')
    ax.annotate(f'NIR RMSE: {np.nanmean(nir_rmse):.2f} m', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=20, color='blue')

    ax.set_xlabel('Stereo Depth (m)', fontsize=22)
    ax.set_ylabel('MDE Depth (m)', fontsize=22)
    ax.set_title('Stereo v MDE Depth', fontsize=25)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.plot([0, 80], [0, 80], 'k--')  # line y=x for reference
    ax.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(output_path + 'RGBvNIR_allPoints_surf.png', dpi=180)
    plt.close('all')


def histogram_stack(data_folder:str, output_path:str):

    #stack histograms of RGB and NIR
    # overlay RGB and NIR for one to one plots
    RGB_dict_list = load_data(cam='RGB', dataset=None, data_folder=data_folder)
    # rename dictionary keys
    for d in RGB_dict_list:
        d['name'] = d['name'] + '_rgb'

    NIR_dict_list = load_data(cam='NIR', dataset=None, data_folder=data_folder)

    dicts = comp_timestamps(RGB_dict_list + NIR_dict_list)

    # background images
    rgb_image = dicts['aux_image_rect_color_rgb']['messages']  # RGB image
    nir_image = dicts['left_image_rect']['messages']  # NIR image

    # mde predictions
    rgb_mde_depth = dicts['aux_image_rect_color_mde_rgb']['messages']  # RGB MDE Depth
    nir_mde_depth = dicts['left_image_rect_mde']['messages']  # NIR MDE depth

    # stereo depth maps
    rgb_stereo_depth = dicts['left_depth_rgb']['messages']  # RGB StereoDepth
    nir_stereo_depth = dicts['left_depth']['messages']  # NIR StereoDepth

    # create masks above horizon, pull horizon line
    rgb_masks = []
    nir_masks = []
    for i in range(len(rgb_image)):
        rgb_mask, _ = mask_horizon(rgb_image[i])
        rgb_masks.append(rgb_mask)
        nir_mask, _ = mask_horizon(nir_image[i])
        nir_masks.append(nir_mask)
    rgb_masks = np.array(rgb_masks)
    nir_masks = np.array(nir_masks)

    # MDE prediction is run off of RGB image. Stereo is calculated from monochrome cams. Offset in camera location, must shift to correct for it.
    rgb_stereo_depth = shift_data(rgb_stereo_depth, num_pixels=16)  # Shift stereo depth map to match MDE depth map

    # create copies of depth maps for computation and masking
    rgb_mde_depth_ = rgb_mde_depth.copy()
    rgb_stereo_depth_ = rgb_stereo_depth.copy()
    nir_mde_depth_ = nir_mde_depth.copy()
    nir_stereo_depth_ = nir_stereo_depth.copy()

    # apply mask to depth maps and background image
    if rgb_masks.ndim > 3:
        rgb_masks = rgb_masks[:, :, :, 0]  # Remove the channel dimension if it exists
    for i in range(len(rgb_mde_depth)):
        rgb_mde_depth_[i] = np.where(rgb_masks[i] != 255, rgb_mde_depth_[i], np.nan)
        nir_mde_depth_[i] = np.where(nir_masks[i] != 255, nir_mde_depth_[i], np.nan)
        rgb_stereo_depth[i] = np.where(rgb_masks[i] != 255, rgb_stereo_depth[i], np.nan)
        nir_stereo_depth[i] = np.where(nir_masks[i] != 255, nir_stereo_depth[i], np.nan)

    # create box around the LARC, turn into NaNs
    yy, xx = np.meshgrid(np.arange(rgb_mde_depth_.shape[1]), np.arange(rgb_mde_depth_.shape[2]),
                         indexing='ij')  # RGB and NIR images should be the same size

    rgb_larc_mask = ((xx >= 500) & (yy >= 530)) | (xx >= 800)  # creates 600x960 mask
    rgb_larc_mask = np.broadcast_to(rgb_larc_mask,
                                    rgb_mde_depth_.shape)  # broadcast 2d larc_mask to shape of all depth maps nx600x960
    rgb_mde_depth_[rgb_larc_mask] = np.nan  # apply mask to MDE depth map
    rgb_stereo_depth_[rgb_larc_mask] = np.nan  # apply mask to Stereo depth map

    nir_larc_mask = ((xx >= 370) & (yy >= 400)) | ((620 <= xx) & (xx <= 720) & (yy >= 120))
    nir_larc_mask = np.broadcast_to(nir_larc_mask,
                                    nir_mde_depth_.shape)  # broadcast 2d larc_mask to shape of all depth maps
    nir_mde_depth_[nir_larc_mask] = np.nan  # apply mask to MDE depth map
    nir_stereo_depth_[nir_larc_mask] = np.nan  # apply mask to Stereo depth map

    # mask out values greater than or equal to 80m
    rgb_mde_depth_[rgb_mde_depth_ >= 80] = np.nan
    rgb_stereo_depth_[rgb_stereo_depth_ >= 80] = np.nan
    nir_mde_depth_[nir_mde_depth_ >= 80] = np.nan
    nir_stereo_depth_[nir_stereo_depth_ >= 80] = np.nan

    #difference stereo and mde to plot
    rgb_diff = rgb_mde_depth_ - rgb_stereo_depth_
    nir_diff = nir_mde_depth_ - nir_stereo_depth_

    combined_min = np.nanmin(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
    combined_max = np.nanmax(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
    bins = np.linspace(combined_min, combined_max, 205)  # define edges for consistent bin widths

    # for i in range(len(rgb_mde_depth)):
    #     # create stacked histogram of RGB and NIR depths
    #     fig = plt.figure(figsize=(20, 20))
    #     ax = fig.add_subplot(111)
    #
    #     if np.isnan(rgb_diff[i]).all() and np.isnan(nir_diff[i]).all():
    #         continue #skip histogram if all values are NaN
    #
    #     ax.hist(rgb_diff[i].flatten(), bins=bins, color='orange', edgecolor='black', density=True, alpha=0.5, label='RGB')
    #     ax.hist(nir_diff[i].flatten(), bins=bins, color='cyan', edgecolor='black', density=True, alpha=0.5, label='NIR')
    #
    #     ax.set_xlabel('Error (m)', fontsize=20)
    #     ax.set_ylabel('Probability Density', fontsize=20)
    #     ax.set_title('RGB & NIR MDE v Stereo Depth Error', fontsize=22)
    #     ax.set_xlim(-40, 40)
    #     ax.set_ylim(0, 0.3)
    #     ax.tick_params(axis='both', which='major', labelsize=15)
    #     ax.legend(fontsize=18)
    #
    #     plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
    #     plt.close('all')


    #plot histogram stack for the entire dataset
    rgb_diff_mean = np.nanmean(rgb_diff, axis=(1, 2))
    nir_diff_mean = np.nanmean(nir_diff, axis=(1, 2))

    combined_min = np.nanmin(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
    combined_max = np.nanmax(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
    bins = np.linspace(combined_min, combined_max, 69)  # define edges for consistent bin widths

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)

    ax.hist(rgb_diff_mean, bins=bins, color='orange', edgecolor='black', density=True, alpha=0.5, label='RGB')
    ax.hist(nir_diff_mean, bins=bins, color='cyan', edgecolor='black', density=True, alpha=0.5, label='NIR')

    ax.set_xlabel('Error (m)', fontsize=20)
    ax.set_ylabel('Probability Density', fontsize=20)
    ax.set_title('MDE-Stereo Depth Error Distribution', fontsize=22)
    ax.set_xlim(-15, 15)
    ax.set_ylim(0, 0.3)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(fontsize=18)

    plt.savefig(output_path + 'histogram_stack_allPoints_openwater' + '.png', dpi=180)  #TODO: add a flag to save a plot of entire dataset
    plt.close('all')







