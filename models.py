#define paths for models
import torch
import numpy as np
import os
import sys
import cv2

sys.path.append('./Depth_Anything_V2')
sys.path.append('./Depth_Anything_V2/metric_depth')
sys.path.append('./DPT')
sys.path.append('./GLPDepth')
sys.path.append('./ml_depth_pro')
sys.path.append('./ZoeDepth')

def dep_any(device, pred:str, encoder: str='vitb', dataset: str='vkitti', max_depth: int=80):


    from depth_anything_v2.dpt import DepthAnythingV2
    from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2Metric

    if pred=='relative':
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = encoder

        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'./Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cuda'))
        model = model.to(device).eval()

        return model

    if pred=='metric':
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        encoder = encoder
        dataset = dataset  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth = max_depth # 20 for indoor model, 80 for outdoor model

        model = DepthAnythingV2Metric(**{**model_configs[encoder], 'max_depth': max_depth})
        model.load_state_dict(
            torch.load(f'./Depth_Anything_V2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cuda'))
        model.to(device).eval()

        return model


def dep_pro(device: str='cuda'):


    from ml_depth_pro.src import depth_pro

    model, transform = depth_pro.create_model_and_transforms(device=torch.device(device))
    model.eval()

    return model, transform


def intel_dpt(device: str='cuda', model_path: str='./DPT/weights/dpt_hybrid_kitti-cb926ef4.pt', optimize=True):

    from DPT.dpt.models import DPTDepthModel
    from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose


    net_w = 1216
    net_h = 352

    model = DPTDepthModel(
        path=model_path,
        scale=0.00006016,
        shift=0.00579,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False
    )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    return model, transform


def glpn(device: str='cuda', max_depth: int=80, checkpoint: str='./GLPDepth/checkpoints/best_model_kitti.ckpt'):

    import torch.backends.cudnn as cudnn
    from GLPDepth.code.models.model import GLPDepth
    from collections import OrderedDict

    if device == 'cuda':
        cudnn.benchmark = True

    model = GLPDepth(max_depth=max_depth, is_train=False).to(device)
    model_weight = torch.load(checkpoint)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    return model


def intel_zoe(device: str='cuda'):


    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config

    config = get_config("zoedepth_nk", "infer") #ZoeDepth nk for metric
    config['pretrained_resource'] = None  #set pretrained_resource to None to avoid mismatch of keys between models
    model = build_model(config)
    ckpt = torch.load("./ZoeDepth/checkpoints/ZoeD_M12_NK.pt", map_location=device) #load model checkpoint and send to GPU
    model.load_state_dict(ckpt["model"], strict=False)

    return model
