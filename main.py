import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import models
import depth_utils
import re
import depth_plots


if __name__ == "__main__":

    # select device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    '''SELECT MODEL'''
    # model = models.dep_any(device, pred='metric')
    # model = models.glpn(device)
    # model = models.intel_zoe(device)
    # model, transform = models.dep_pro(device)

    #MODEL TYPES: 'dep_any', 'glpn', 'dpt_zoe', 'dep_pro'

    '''RUN VIDEO'''
    # input_path = glob.glob('./tower_images/video/*.MOV')
    # data_name = [os.path.basename(path).split('.')[0] for path in input_path]  # Extract file names without extensions
    # depth_utils.run_video(model, input_path, device, run_model='dep_any', fps=1, data_name=data_name)

    '''RUN TOWERFRAMES'''
    # input_path = '/mnt/e/towerframes'
    # data_name = [re.match(r'^(\d+)', os.path.basename(path)).group(1) for path in os.listdir(input_path)] #pull timestamps from towerframes folder
    # depth_utils.run_towerframes(model, input_path, device, gcp=True, run_model='dep_any', data_name=data_name) #only takes the first 31 images for efficiency. Go into create input function to change.

    '''RUN RAW IMAGES'''
    # input_path = ['./data/ms_output_1/ms2/left_image_rect/data.npz', './data/ms_output_1/ms2/right_image_rect/data.npz', './data/ms_output_1/ms2/aux_image_rect_color/data.npz']
    # data_name = [f'{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))}_{os.path.basename(os.path.dirname(os.path.dirname(path)))}_{os.path.basename(os.path.dirname(path))}' for path in input_path]
    # depth_utils.run_rawImages(model, input_path, device, run_model='dep_any', fps=1, data_name=data_name)

    '''PLOT DATA'''
    depth_plots.histogram(data_folder='./Depth_Anything_V2/data', output_path='./Depth_Anything_V2/outputs/temp/', flag='RGB', dataset='ms_output_1', all_data=True)










