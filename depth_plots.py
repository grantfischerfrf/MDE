import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec


def four_panel_plot(sp1, sp2, sp3, sp4, img_path:str, output_path:str):

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

    plt.tight_layout()
    plt.savefig(output_path + os.path.basename(img_path).split('.')[0] + '_velocity_map.png', dpi=200)
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


def mean_depth(sp1, sp2, UV=None, ind=None, mean_est_depth=None, gcp=False):

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    depth = ax.imshow(sp1, cmap='tab20', alpha=0.75, aspect='auto', vmin=0, vmax=80)
    ax.imshow(sp2, alpha=0.35, aspect='auto')
    if gcp:
        ax.plot(UV[0], UV[1], 'r+', markersize=8)
        for i in range(len(UV[0])):
            ax.annotate(f'{ind[i]}' + '-' + f'{mean_est_depth[i]:.2f} m', xy=(UV[0][i], UV[1][i]), color='white', fontsize=10, fontweight='bold')

    ax.set_title('Mean Depth Map', fontsize=22)

    cbar = fig.colorbar(depth, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, ticks=np.arange(0, 84, 4)[::2])
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Est. Depth (m)')

    plt.savefig(f'./outputs/mean_depth_map')
    plt.close('all')


def mean_velocity(sp1, sp2, UV=None, ind=None, mean_est_vel=None, gcp=False):

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    velocity = ax.imshow(sp1, cmap='coolwarm', alpha=0.85, aspect='auto', vmin=-1, vmax=1)
    ax.imshow(sp2, alpha=0.35, aspect='auto')
    if gcp:
        ax.plot(UV[0], UV[1], 'r+', markersize=8)
        for i in range(len(UV[0])):
            ax.annotate(f'{ind[i]}' + '-' + f'{mean_est_vel[i]:.2f} m/s', xy=(UV[0][i], UV[1][i]), color='white', fontsize=10, fontweight='bold')
    ax.set_title('Mean Velocity Map', fontsize=22)

    cbar = fig.colorbar(velocity, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, ticks=np.arange(0, 84, 4)[::2])
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Est. Velocity (m/s)')

    plt.savefig(f'./outputs/mean_velocity_map')
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










