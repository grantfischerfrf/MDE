import torch
import numpy as np
import os
import sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def SFML(device:str='cuda'):

    sys.path.append('./ThermalMonoDepth')
    sys.path.append('./ThermalMonoDepth/common')

    from common.models import DispResNet
    model = DispResNet(num_layers=18, num_channel=1)
    weights = torch.load('./ThermalMonoDepth/checkpoints/outdoor_model/dispnet_disp_model_best.pth.tar')
    model.load_state_dict(weights['state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    img_path = './tower_images/video/20250508F01_SRH701384881_IR_0007_reverseTransit/20250508F01_SRH701384881_IR_0007_reverseTransit_2947.jpeg'
    img = torch.tensor(cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)).unsqueeze(0).float().to(device)

    #add batch dimension
    img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        dep_map = model(img)

    # dep_map = (255 * (0.45 + dep_map.squeeze().detach().cpu().numpy() * 0.225)).astype(np.uint8)
    dep_map = dep_map.squeeze().cpu().numpy()

    return model, dep_map


def MonoTherm(device:str='cuda'):

    sys.path.append('./monother_depth')

    import json
    from monother_depth.zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
    from monother_depth.zoedepth.models.builder import build_model

    config = get_config('zoedepth', mode='infer', config_version='ms2_thermal')

    model = build_model(config)
    model = model.to(device)
    model.eval()

    img_path = './tower_images/video/20250508F01_SRH701384881_IR_0007_reverseTransit/20250508F01_SRH701384881_IR_0007_reverseTransit_2947.jpeg'
    # img_path = './tower_images/video/nobeach_pier/nobeach_pier_276.jpeg'
    dep_map = model.infer_pil(Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM))

    return model, dep_map, config



if __name__ == '__main__':

    image = cv2.imread('./tower_images/video/20250508F01_SRH701384881_IR_0007_reverseTransit/20250508F01_SRH701384881_IR_0007_reverseTransit_2947.jpeg')
    # image = cv2.imread('./tower_images/video/nobeach_pier/nobeach_pier_276.jpeg')

    # model, dep_map = SFML()
    #
    model, dep_map, config = MonoTherm()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(121)
    ax.imshow(dep_map, cmap='plasma')
    ax.set_title('Depth Map', fontsize=20)
    ax2 = fig.add_subplot(122)
    ax2.imshow(image)
    ax2.set_title('Image', fontsize=20)

    #add colorbar
    cbar = fig.colorbar(ax.imshow(dep_map, cmap='plasma'), ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    plt.close('all')
