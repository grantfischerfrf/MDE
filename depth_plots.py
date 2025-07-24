from depth_utils import comp_timestamps, find_ref_dict
from plot_utils import load_data, read_dict, clip_data, bin_data

import cv2
import sys
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




def panel_plot(data_folder:str, output_path:str, dataset:str):

    RGB_dict_list = load_data(cam='RGB', dataset=dataset, data_folder=data_folder)
    # rename dictionary keys
    for d in RGB_dict_list:
        d['name'] = d['name'] + '_rgb'

    NIR_dict_list = load_data(cam='NIR', dataset=dataset, data_folder=data_folder)

    dicts = comp_timestamps(RGB_dict_list + NIR_dict_list)

    rgb_images, rgb_mde_depths, rgb_stereo_depths, nir_images, nir_mde_depths, nir_stereo_depths = read_dict('BOTH', dicts)

    rgb_mde_depths_, rgb_stereo_depths_, rgb_horizons = clip_data(rgb_images, rgb_mde_depths, rgb_stereo_depths, 'RGB')
    nir_mde_depths_, nir_stereo_depths_, nir_horizons = clip_data(nir_images, nir_mde_depths, nir_stereo_depths, 'NIR')

    rgb_X, rgb_mde_mean, rgb_mde_std_dev = bin_data(rgb_stereo_depths_, rgb_mde_depths_, n_bins=81)
    nir_X, nir_mde_mean, nir_mde_std_dev = bin_data(nir_stereo_depths_, nir_mde_depths_, n_bins=81)

    # difference stereo and mde to plot
    rgb_diff = rgb_mde_depths_ - rgb_stereo_depths_
    nir_diff = nir_mde_depths_ - nir_stereo_depths_

    combined_min = np.nanmin(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
    combined_max = np.nanmax(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
    bins = np.linspace(combined_min, combined_max, 205)  # define edges for consistent bin widths

    print('Plotting...')
    for i in range(len(rgb_mde_depths)):
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 3)

        #mde map
        ax1 = fig.add_subplot(gs[0, 0])
        plot1 = ax1.imshow(rgb_mde_depths[i], cmap='viridis', alpha=0.75, aspect='equal', vmin=0, vmax=80)
        ax1.imshow(rgb_images[i], alpha=0.35, aspect='equal')
        ax1.plot([rgb_horizons[i][0], rgb_horizons[i][1]], [rgb_horizons[i][2], rgb_horizons[i][3]], color='red',linewidth=3)
        ax1.set_title('RGB MDE Depth Map', fontsize=22)
        ax1.set_xticks([])
        ax1.set_yticks([])

        #rgb stereo map
        ax1b = fig.add_subplot(gs[0, 1])
        plot1b = ax1b.imshow(rgb_stereo_depths[i], cmap='viridis', alpha=0.75, aspect='auto', vmin=0, vmax=80)
        ax1b.imshow(rgb_images[i], alpha=0.35, aspect='equal')
        ax1b.plot([rgb_horizons[i][0], rgb_horizons[i][1]], [rgb_horizons[i][2], rgb_horizons[i][3]], color='red',linewidth=3)
        ax1b.set_title('RGB Stereo Depth Map', fontsize=22)
        ax1b.set_xticks([])
        ax1b.set_yticks([])

        #NIR MDE map
        ax2 = fig.add_subplot(gs[1, 0])
        plot2 = ax2.imshow(nir_mde_depths[i], cmap='viridis', alpha=0.75, aspect='auto', vmin=0, vmax=80)
        ax2.imshow(nir_images[i], alpha=0.35, aspect='equal')
        ax2.plot([nir_horizons[i][0], nir_horizons[i][1]], [nir_horizons[i][2], nir_horizons[i][3]], color='red',linewidth=3)
        ax2.set_title('NIR MDE Depth Map', fontsize=22)
        ax2.set_xticks([])
        ax2.set_yticks([])

        #NIR stereo map
        ax2b = fig.add_subplot(gs[1, 1])
        plot2b = ax2b.imshow(nir_stereo_depths[i], cmap='viridis', alpha=0.75, aspect='auto', vmin=0, vmax=80)
        ax2b.imshow(nir_images[i], alpha=0.35, aspect='equal')
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

        # #calculate difference in timestamps between RGB and NIR image
        # rgb_timestamp = dicts['aux_image_rect_color_rgb']['timestamps'][i]
        # nir_timestamp = dicts['left_image_rect']['timestamps'][i]
        # time_diff = abs(rgb_timestamp - nir_timestamp)

        plt.tight_layout()
        plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
        plt.close('all')


def panel_plot_allData(data_folder:str, output_path:str, dataset:str):

    RGB_dict_list = load_data(cam='RGB', dataset=dataset, data_folder=data_folder)
    # rename dictionary keys
    for d in RGB_dict_list:
        d['name'] = d['name'] + '_rgb'

    NIR_dict_list = load_data(cam='NIR', dataset=dataset, data_folder=data_folder)

    dicts = comp_timestamps(RGB_dict_list + NIR_dict_list)

    rgb_images, rgb_mde_depths, rgb_stereo_depths, nir_images, nir_mde_depths, nir_stereo_depths = read_dict('BOTH', dicts)

    rgb_mde_depths_, rgb_stereo_depths_, rgb_horizons = clip_data(rgb_images, rgb_mde_depths, rgb_stereo_depths, 'RGB')
    nir_mde_depths_, nir_stereo_depths_, nir_horizons = clip_data(nir_images, nir_mde_depths, nir_stereo_depths, 'NIR')

    rgb_X, rgb_mde_mean, rgb_mde_std_dev = bin_data(rgb_stereo_depths_, rgb_mde_depths_, n_bins=81)
    nir_X, nir_mde_mean, nir_mde_std_dev = bin_data(nir_stereo_depths_, nir_mde_depths_, n_bins=81)

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2)

    # rgb mde map
    ax1 = fig.add_subplot(gs[0, 0])
    plot1 = ax1.imshow(rgb_mde_depths[-1], cmap='viridis', alpha=0.75, aspect='auto', vmin=0, vmax=80)
    ax1.imshow(rgb_images[-1], alpha=0.35, aspect='equal')
    ax1.plot([rgb_horizons[-1][0], rgb_horizons[-1][1]], [rgb_horizons[-1][2], rgb_horizons[-1][3]], color='red', linewidth=3)
    ax1.set_title('RGB MDE Depth Map', fontsize=22)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # nir mde map
    ax2 = fig.add_subplot(gs[0, 1])
    plot2 = ax2.imshow(nir_mde_depths[-1], cmap='viridis', alpha=0.75, aspect='auto', vmin=0, vmax=80)
    ax2.imshow(nir_images[-1], alpha=0.35, aspect='equal')
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
    rgb_diff = rgb_mde_depths_ - rgb_stereo_depths_
    nir_diff = nir_mde_depths_ - nir_stereo_depths_
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
    plt.savefig(output_path + 'DepthComp_allData' + '.png', dpi=180)
    plt.close('all')


def depth_comp(data_folder:str, output_path:str, flag:str, dataset:str):

    dict_list = load_data(cam=flag, dataset=dataset, data_folder=data_folder)

    dicts = comp_timestamps(dict_list)

    images, mde_depths, stereo_depths = read_dict(flag, dicts)

    mde_depths_, stereo_depths_, horizons = clip_data(images, mde_depths, stereo_depths, flag)

    abs_diff = np.abs(stereo_depths_ - mde_depths_) # Absolute Difference between MDE and StereoDepth

    rmse, rmse_10m, rmse_20m, rmse_40m, rmse_p, rmse_10p, rmse_20p, rmse_40p = cal_rmses(mde_depths_, stereo_depths_, abs_diff)

    for i in range(len(images)):
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        plot1 = ax1.imshow(mde_depths[i], cmap='viridis', alpha=0.75, aspect='equal', vmin=0, vmax=80)
        ax1.imshow(images[i], alpha=0.35, aspect='equal')
        ax1.plot([horizons[i][0], horizons[i][1]], [horizons[i][2], horizons[i][3]], color='red', linewidth=3)
        ax1.set_title(f'{flag} MDE Depth Map', fontsize=22)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[0, 1])
        plot2 = ax2.imshow(stereo_depths[i], cmap = 'viridis', alpha=0.75, aspect='equal', vmin=0, vmax=80)  #Depth map is getting clipped for some reason
        ax2.imshow(images[i], alpha=0.35, aspect='equal')
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
        plot3 = ax3.imshow(abs_diff[i], cmap='Greys', aspect='equal', vmin=0, vmax=10)
        ax3.set_title(f'Difference (abs value)', fontsize=22)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ins_ax3 = inset_axes(ax3, width='25%', height='25%', loc='upper left', bbox_to_anchor=(0.08, 0.23, .75, .75), bbox_transform=ax3.transAxes)
        ins_ax3.plot([10, 20, 40, 80], [rmse_10m[i], rmse_20m[i], rmse_40m[i], rmse[i]], marker='o', color='blue', linewidth=2, markersize=5, label='RMSE')
        ins_ax3.set_xticks([10, 20, 40, 80])
        ins_ax3.set_ylim(0,10)
        ins_ax3.tick_params(axis='x', labelrotation=45, labelsize=8)
        ins_ax3.set_xlabel('Depth (m)')
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
        ax4.imshow(images[i], aspect='equal')
        ax4.plot([horizons[i][0], horizons[i][1]], [horizons[i][2], horizons[i][3]], color='red', linewidth=3)
        # if flag == 'RGB':
        #     ax4.plot([500, 500], [599, 530], color='red', linewidth=3)
        #     ax4.plot([500, 800], [530, 530], color='red', linewidth=3)
        #     ax4.plot([800, 800], [530, 0], color='red', linewidth=3)
        # if flag == 'NIR':
        #     ax4.plot([370, 370], [599, 400], color='red', linewidth=3)
        #     ax4.plot([370, 620], [400, 400], color='red', linewidth=3)
        #     ax4.plot([620, 620], [400, 120], color='red', linewidth=3)
        #     ax4.plot([620, 720], [120, 120], color='red', linewidth=3)
        #     ax4.plot([720, 720], [400, 120], color='red', linewidth=3)
        #     ax4.plot([720, 959], [400, 400], color='red', linewidth=3)
        ax4.set_title(f'{flag} Image', fontsize=22)
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

        fig.suptitle(f'Depth Comparison: {flag} MDE vs Stereo Depth', fontsize=24, fontweight='bold')

        plt.tight_layout()

        if flag == 'RGB':
            plt.savefig(output_path + str(dicts['aux_image_rect_color']['timestamps'][i]) + '.png', dpi=180)
        elif flag == 'NIR':
            plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
        else:
            raise ValueError(f"Unknown flag: {flag}. Options are 'RGB', 'NIR', or 'BOTH'.")

        plt.close('all')


def one_to_one(data_folder:str, output_path:str, flag:str, dataset:str, all_data:bool=False):

    dict_list = load_data(cam=flag, dataset=dataset, data_folder=data_folder)

    dicts = comp_timestamps(dict_list)

    images, mde_depths, stereo_depths = read_dict(flag, dicts)

    mde_depths_, stereo_depths_, _ = clip_data(images, mde_depths, stereo_depths, flag)

    rmse = np.sqrt(np.nanmean((stereo_depths_ - mde_depths_) ** 2, axis=(1, 2)))  # Calculate RMSE for each depth map

    if not all_data:
        #plot one to one plot of stereo and mde depth maps
        for i in range(len(mde_depths_)):
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111)

            ax.plot(stereo_depths_[i], mde_depths_[i], 'ro', markersize=5, alpha=0.1)
            ax.set_xlabel('Stereo Depth (m)', fontsize=15)
            ax.set_ylabel('MDE Depth (m)', fontsize=15)
            ax.set_title(f'Stereo vs MDE Depth: RMSE = {rmse[i]:.2f}m', fontsize=22)
            ax.plot([0, 80], [0, 80], 'k--')  # line y=x for reference
            ax.set_xlim(0, 80)
            ax.set_ylim(0, 80)

            plt.tight_layout()
            if flag == 'RGB':
                plt.savefig(output_path + str(dicts['aux_image_rect_color']['timestamps'][i]) + '.png', dpi=180)
            elif flag == 'NIR':
                plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
            else:
                raise ValueError(f"Unknown flag: {flag}. Options are 'RGB', 'NIR', or 'BOTH'.")

            plt.close('all')

    if all_data:
        # plot scatter plot of all points
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        for i in range(len(mde_depths_)):
            ax.plot(stereo_depths_[i][::10, ::10], mde_depths_[i][::10, ::10], 'r+', markersize=3, alpha=0.1) #scatter plot
        ax.set_xlabel('Stereo Depth (m)', fontsize=15)
        ax.set_ylabel('MDE Depth (m)', fontsize=15)
        ax.set_title(f'Stereo vs MDE Depth: RMSE = {np.mean(rmse):.2f}m', fontsize=22)
        ax.plot([0, 80], [0, 80], 'k--')  # line y=x for reference
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 80)

        plt.savefig(output_path + f'{flag}_{dataset}_all_points.png', dpi=180)
        plt.close('all')


def histogram(data_folder:str, output_path:str, flag:str, dataset:str=None):

    dict_list = load_data(cam=flag, dataset=dataset, data_folder=data_folder)

    dicts = comp_timestamps(dict_list)

    images, mde_depths, stereo_depths = read_dict(flag, dicts)

    mde_depths_, stereo_depths_, _ = clip_data(images, mde_depths, stereo_depths, flag)

    for i in range(len(mde_depths_)):
        #create histogram of all points
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        ax.hist(mde_depths_[i].flatten() - stereo_depths_[i].flatten(), bins=68, color='blue', edgecolor='black', density=True)
        ax.set_xlabel('Error (m)', fontsize=20)
        ax.set_ylabel('Probability Density', fontsize=20)
        ax.set_title('RGB MDE v Stereo Depth Error', fontsize=22)
        ax.set_xlim(-40, 40)
        ax.set_ylim(0, 0.4)
        ax.tick_params(axis='both', which='major', labelsize=15)

        if flag == 'RGB':
            plt.savefig(output_path + str(dicts['aux_image_rect_color']['timestamps'][i]) + '.png', dpi=180)
        elif flag == 'NIR':
            plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
        else:
            raise ValueError(f"Unknown flag: {flag}. Options are 'RGB', 'NIR', or 'BOTH'.")

        plt.close('all')


def heatmap(data_folder:str, output_path:str, flag:str, dataset:str=None, all_data:bool=False):

    dict_list = load_data(cam=flag, dataset=dataset, data_folder=data_folder)

    dicts = comp_timestamps(dict_list)

    images, mde_depths, stereo_depths = read_dict(flag, dicts)

    mde_depths_, stereo_depths_, _ = clip_data(images, mde_depths, stereo_depths, flag)

    if not all_data:
        # plot scatter density/heatmap of all points
        for i in range(len(mde_depths_)):

            #filter out nans
            mde_depth = mde_depths_[i][(~np.isnan(mde_depths_[i])) & (~np.isnan(stereo_depths_[i]))]
            stereo_depth = stereo_depths_[i][(~np.isnan(mde_depths_[i])) & (~np.isnan(stereo_depths_[i]))]

            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111)
            ax.hist2d(stereo_depth.flatten(), mde_depth.flatten(), bins=68, cmap='hot_r', norm=colors.LogNorm(), range=((0, 80), (0, 80)))
            ax.set_xlabel('Stereo Depth (m)', fontsize=20)
            ax.set_ylabel('MDE Depth (m)', fontsize=20)
            ax.set_title('Stereo v MDE depth Heatmap', fontsize=22)
            ax.set_xlim(0, 80)
            ax.set_ylim(0, 80)
            ax.plot([0, 80], [0, 80], color='magenta', linestyle='--')  # line y=x for reference
            ax.tick_params(axis='both', which='major', labelsize=15)

            #colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(ax.collections[0], cax=cax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=15)
            cbar.set_label('Count', fontsize=20)

            if flag == 'RGB':
                plt.savefig(output_path + str(dicts['aux_image_rect_color']['timestamps'][i]) + '.png', dpi=180)
            elif flag == 'NIR':
                plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
            else:
                raise ValueError(f"Unknown flag: {flag}. Options are 'RGB', 'NIR', or 'BOTH'.")

            plt.close('all')

    if all_data:
        #create heatmap for total dataset
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)

        mde_depth = mde_depths_[(~np.isnan(mde_depths_)) & (~np.isnan(stereo_depths_))]
        stereo_depth = stereo_depths_[(~np.isnan(mde_depths_)) & (~np.isnan(stereo_depths_))]

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

        plt.savefig(output_path + f'{flag}_{dataset}_total_heatmap.png', dpi=180)
        plt.close('all')


def fill_plot(data_folder:str, output_path:str, flag:str, dataset:str=None):

    dict_list = load_data(cam=flag, dataset=dataset, data_folder=data_folder)

    dicts = comp_timestamps(dict_list)

    images, mde_depths, stereo_depths = read_dict(flag, dicts)

    mde_depths_, stereo_depths_, _ = clip_data(images, mde_depths, stereo_depths, flag)

    X, mde_mean, mde_std_dev = bin_data(stereo_depths_, mde_depths_, n_bins=81)

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

        if flag == 'RGB':
            plt.savefig(output_path + str(dicts['aux_image_rect_color']['timestamps'][i]) + '.png', dpi=180)
        elif flag == 'NIR':
            plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
        else:
            raise ValueError(f"Unknown flag: {flag}. Options are 'RGB', 'NIR', or 'BOTH'.")

        plt.close('all')


def fill_overlay(data_folder:str, output_path:str, dataset:str, all_data:bool=False):

    RGB_dict_list = load_data(cam='RGB', dataset=dataset, data_folder=data_folder)
    # rename dictionary keys
    for d in RGB_dict_list:
        d['name'] = d['name'] + '_rgb'

    NIR_dict_list = load_data(cam='NIR', dataset=dataset, data_folder=data_folder)

    dicts = comp_timestamps(RGB_dict_list + NIR_dict_list)

    rgb_images, rgb_mde_depths, rgb_stereo_depths, nir_images, nir_mde_depths, nir_stereo_depths = read_dict('BOTH', dicts)

    rgb_mde_depths_, rgb_stereo_depths_, _ = clip_data(rgb_images, rgb_mde_depths, rgb_stereo_depths, 'RGB')
    nir_mde_depths_, nir_stereo_depths_, _ = clip_data(nir_images, nir_mde_depths, nir_stereo_depths, 'NIR')

    rgb_X, rgb_mde_mean, rgb_mde_std_dev = bin_data(rgb_stereo_depths_, rgb_mde_depths_, n_bins=81)
    nir_X, nir_mde_mean, nir_mde_std_dev = bin_data(nir_stereo_depths_, nir_mde_depths_, n_bins=81)

    print('Plotting...')
    # create fill between plot
    if not all_data:
        for i in range(np.array(rgb_mde_mean).shape[1]):
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111)

            rgb_mde_mean_i = np.array(rgb_mde_mean)[:, i]
            rgb_mde_std_dev_i = np.array(rgb_mde_std_dev)[:, i]
            nir_mde_mean_i = np.array(nir_mde_mean)[:, i]
            nir_mde_std_dev_i = np.array(nir_mde_std_dev)[:, i]

            ax.plot(rgb_X, rgb_mde_mean_i + rgb_mde_std_dev_i, color='red', linestyle='--', linewidth=2, label='+ 1 Standard Deviation')
            ax.plot(rgb_X, rgb_mde_mean_i - rgb_mde_std_dev_i, color='blue', linestyle='--', linewidth=2,
                    label='- 1 Standard Deviation')
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

            plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
            plt.close('all')

    #plot fill overlay for the entire dataset
    if all_data:
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


def one_to_one_overlay(data_folder:str, output_path:str, dataset:str):

    RGB_dict_list = load_data(cam='RGB', dataset=dataset, data_folder=data_folder)
    # rename dictionary keys
    for d in RGB_dict_list:
        d['name'] = d['name'] + '_rgb'

    NIR_dict_list = load_data(cam='NIR', dataset=dataset, data_folder=data_folder)

    dicts = comp_timestamps(RGB_dict_list + NIR_dict_list)

    rgb_images, rgb_mde_depths, rgb_stereo_depths, nir_images, nir_mde_depths, nir_stereo_depths = read_dict('BOTH', dicts)

    rgb_mde_depths_, rgb_stereo_depths_, _ = clip_data(rgb_images, rgb_mde_depths, rgb_stereo_depths, 'RGB')
    nir_mde_depths_, nir_stereo_depths_, _ = clip_data(nir_images, nir_mde_depths, nir_stereo_depths, 'NIR')

    rgb_rmse = np.sqrt(np.nanmean((rgb_stereo_depths_ - rgb_mde_depths_)**2, axis=(1, 2)))
    nir_rmse = np.sqrt(np.nanmean((nir_stereo_depths_ - nir_mde_depths_)**2, axis=(1, 2)))

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    print('plotting...')
    for i in range(len(rgb_mde_depths_)):
        ax.plot(rgb_stereo_depths_[i][::20, ::20], rgb_mde_depths_[i][::20, ::20],'r+', markersize=3, alpha=0.1, label='RGB')
        ax.plot(nir_stereo_depths_[i][::20, ::20], nir_mde_depths_[i][::20, ::20],'b+', markersize=3, alpha=0.1, label='NIR')

    ax.annotate(f'RGB RMSE: {np.nanmean(rgb_rmse):.2f} m', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=20, color='red')
    ax.annotate(f'NIR RMSE: {np.nanmean(nir_rmse):.2f} m', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=20, color='blue')

    ax.set_xlabel('Stereo Depth (m)', fontsize=22)
    ax.set_ylabel('MDE Depth (m)', fontsize=22)
    ax.set_title('Stereo v MDE Depth', fontsize=25)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.plot([0, 80], [0, 80], 'k--')  # line y=x for reference
    ax.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(output_path + 'RGB v NIR_allPoints.png', dpi=180)
    plt.close('all')


def histogram_stack(data_folder:str, output_path:str, dataset:str, all_data:bool=False):

    RGB_dict_list = load_data(cam='RGB', dataset=dataset, data_folder=data_folder)
    # rename dictionary keys
    for d in RGB_dict_list:
        d['name'] = d['name'] + '_rgb'

    NIR_dict_list = load_data(cam='NIR', dataset=dataset, data_folder=data_folder)

    dicts = comp_timestamps(RGB_dict_list + NIR_dict_list)

    rgb_images, rgb_mde_depths, rgb_stereo_depths, nir_images, nir_mde_depths, nir_stereo_depths = read_dict('BOTH', dicts)

    rgb_mde_depths_, rgb_stereo_depths_, _ = clip_data(rgb_images, rgb_mde_depths, rgb_stereo_depths, 'RGB')
    nir_mde_depths_, nir_stereo_depths_, _ = clip_data(nir_images, nir_mde_depths, nir_stereo_depths, 'NIR')

    #difference stereo and mde to plot
    rgb_diff = rgb_mde_depths_ - rgb_stereo_depths_
    nir_diff = nir_mde_depths_ - nir_stereo_depths_

    if not all_data:
        combined_min = np.nanmin(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
        combined_max = np.nanmax(np.concatenate([rgb_diff.flatten(), nir_diff.flatten()]))
        bins = np.linspace(combined_min, combined_max, 69)  # define edges for consistent bin widths

        for i in range(len(rgb_mde_depths_)):
            # create stacked histogram of RGB and NIR depths
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111)

            if np.isnan(rgb_diff[i]).all() and np.isnan(nir_diff[i]).all():
                continue #skip histogram if all values are NaN

            ax.hist(rgb_diff[i].flatten(), bins=bins, color='orange', edgecolor='black', density=True, alpha=0.5, label='RGB')
            ax.hist(nir_diff[i].flatten(), bins=bins, color='cyan', edgecolor='black', density=True, alpha=0.5, label='NIR')

            ax.set_xlabel('Error (m)', fontsize=20)
            ax.set_ylabel('Probability Density', fontsize=20)
            ax.set_title('RGB & NIR MDE v Stereo Depth Error', fontsize=22)
            ax.set_xlim(-40, 40)
            ax.set_ylim(0, 0.3)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.legend(fontsize=18)

            plt.savefig(output_path + str(dicts['left_image_rect']['timestamps'][i]) + '.png', dpi=180)
            plt.close('all')

    if all_data:
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

        plt.savefig(output_path + 'histogram_stack_allPoints' + '.png', dpi=180)
        plt.close('all')







def four_panel_plot(sp1, sp2, sp3, sp4, img_path: str, output_path: str, title: str):

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

    cbar = fig.colorbar(plot2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04,
                        ticks=np.arange(0, 84, 4)[::2])
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
        ax3.annotate(f'{ind[i]}: {UV_vel[i]:.2f} m/s', xy=(UV[0][i], UV[1][i]), color='white', fontsize=10,
                     fontweight='bold')
    ax3.set_title('Instantaneous Velocity (rollingAvg)', fontsize=22)
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = fig.add_subplot(gs[1, 1])
    plot4 = ax4.imshow(sp4, cmap='OrRd', norm=colors.LogNorm(), aspect='auto')
    ax4.set_title('2* Standard Deviation (m/s)', fontsize=22)
    ax4.set_xticks([])
    ax4.set_yticks([])

    cbar = fig.colorbar(plot2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04,
                        ticks=np.arange(0, 84, 4)[::2])
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
        ax.annotate(f'{int(calculated_depth[i][0])}', xy=(calculated_depth[i][1], estimated_depth[i]),
                    xytext=(4, -1), textcoords='offset points', color='black', fontsize=5.5)
    ax.set_xlabel('Calculated GCP Depth (m)')
    ax.set_ylabel('Estimated Depth (m)')
    ax.set_title(
        f'{str(date.year)}{date.month:02d}{date.day:02d}{camera} Est v Cal GCP Depths: RMSE = {rmse:.2f}m')
    ax.plot([0, vmax], [0, vmax], 'k--')  # line y=x for reference
    plt.savefig(
        f'{os.path.dirname(output_path.rstrip('/'))}/outputs/gcp/{str(date.year)}{date.month:02d}{date.day:02d}_{camera}' + '_gcp_error.png',
        dpi=180)
    # plt.show()
    plt.close('all')


def error_comparison(est_depths: list, cal_depths: list, labels: list, rmse: list, output_path, vmax=80):

    # est_depths, cal_depths should be a list of lists
    # labels should be a list of strings

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    for i in range(len(labels)):
        ax.scatter(cal_depths[i], est_depths[i], s=20, marker='o', label=labels[i])

    ax.set_xlabel('Calculated GCP Depth (m)')
    ax.set_ylabel('Estimated Depth (m)')
    ax.set_title('Estimated vs Calculated GCP Depths')
    ax.plot([0, vmax], [0, vmax], 'k--')  # line y=x for reference
    ax.text(0.05, 0.95, f'Bob RMSE: {rmse[0]:.2f}m\nMary RMSE: {rmse[1]:.2f}m', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend()
    plt.savefig(f'{os.path.dirname(output_path.rstrip('/'))}/outputs/gcp/allData' + '_gcp_depth_comparison.png',
                dpi=180)
    plt.close('all')










