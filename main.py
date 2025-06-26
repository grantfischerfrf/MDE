import torch
import models
import depth_utils
import depth_plots


if __name__ == "__main__":

    # select device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # select model
    # model = models.dep_any(device, pred='metric')
    model = models.glpn(device)
    # model = models.intel_zoe(device)
    # model, transform = models.dep_pro(device)

    # IR data
    input_path = ['/mnt/e/surrogate_lwir_data/skyraiderR80D/fov_offshore/20250508F01_SRH701384881_IR_0007_reverseTransit.TS']
    # depth_utils.processVideo(input_path[0], create_frames=True)

    '''RUN VIDEO'''
    # input_path = glob.glob('./tower_images/video/*.MOV')
    depth_utils.run_video(model, input_path, device, run_model='glpn', fps=2)

    '''RUN TOWERFRAMES'''
    # input_path = '/mnt/e/towerframes'
    # depth_utils.run_towerframes(model, input_path, device, run_model='dep_pro', gcp=True)