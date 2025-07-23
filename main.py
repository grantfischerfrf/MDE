import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import models
import depth_utils
import depth_plots



if __name__ == "__main__":

    # select device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    '''SELECT MODEL'''
    # model = models.dep_any(device, pred='metric')
    # model = models.glpn(device)
    # model = models.intel_zoe(device)
    # model, transform = models.dep_pro(device)

    # LWIR data
    # input_path = ['/mnt/e/surrogate_lwir_data/skyraiderR80D/fov_offshore/20250508F01_SRH701384881_IR_0007_reverseTransit.TS']
    # depth_utils.processVideo(input_path[0], create_frames=True)

    #MODEL TYPES: 'dep_any', 'glpn', 'dpt_zoe', 'dep_pro'

    '''RUN VIDEO'''
    # input_path = glob.glob('./tower_images/video/*.MOV')
    # data_name='video'
    # depth_utils.run_video(model, input_path, device, run_model='glpn', fps=1, data_name=data_name)

    '''RUN TOWERFRAMES'''
    # input_path = '/mnt/e/towerframes'
    # data_name = 'towerframes' #probably use datetime here
    # depth_utils.run_towerframes(model, input_path, device, gcp=False, run_model='dep_any', data_name=data_name)

    '''RUN RAW IMAGES'''
    # input_path = ['/mnt/e/ms_output_2/ms2_surf/left_image_rect/data.npz', '/mnt/e/ms_output_2/ms2_surf/right_image_rect/data.npz', '/mnt/e/ms_output_2/ms2_surf/aux_image_rect_color/data.npz']
    # data_name = [f'{os.path.basename(os.path.dirname(os.path.dirname(path)))}_{os.path.basename(os.path.dirname(path))}' for path in input_path]
    # depth_utils.run_rawImages(model, input_path, device, run_model='dep_any', fps=1, data_name=data_name)

    '''PLOT DATA'''
    data = depth_plots.panel_plot(data_folder='./Depth_Anything_V2/data', output_path='./Depth_Anything_V2/outputs/temp/')









