import models
import depth_plots
from image_processing import xyz2DistUV
from ml_depth_pro.src import depth_pro
from ZoeDepth.zoedepth.utils.misc import pil_to_batched_tensor

import re
import os
import cv2
import glob
import torch
import pandas as pd
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt


def numerical_sort(string):

    parts = re.split(r'(\d+)', string)
    return [int(part) if part.isdigit() else part for part in parts]


def cal_gcpDis(gcp_xyz, extrinsics, ind):

    # Calculate distance from camera to GCP point
    gcp_dis = np.sqrt((gcp_xyz[0] - extrinsics[0][0]) ** 2 + (gcp_xyz[1] - extrinsics[0][1]) ** 2 + (gcp_xyz[2] - extrinsics[0][2]) ** 2)
    # print(f'GCP-{ind} distance from camera: {gcp_dis}m')
    return gcp_dis


def pull_files(file_path):

    # use datetime to pull all data collect dates from folder
    date_list = os.listdir(file_path)
    date_list = [datetime.strptime(match.group(), '%Y%m%d') for date in date_list if (match := re.match(r'^\d{8}', date))]

    return date_list


def pull_txt(year, month, day):

    if os.path.isfile(f'./txt/{year}{month}{day}'):
        return f'./txt/{year}{month}{day}.txt'
    else:
        txt_list = pull_files('./txt')
        target_date = datetime(int(year), int(month), int(day))
        closest_date = min(txt_list, key=lambda date: abs(date - target_date))
        return f'./txt/{closest_date.year}{closest_date.month:02d}{closest_date.day:02d}.txt'


def pull_ioeo(year, month, day, camera):

    if os.path.isfile(f'./ioeo/{camera}/{year}{month}{day}_{camera}_IOEOInitial.mat'):
        ioeo = scipy.io.loadmat(f'./ioeo/{camera}/{str(year) + str(month) + str(day)}_{camera}_IOEOInitial.mat')
        ioeo = list(ioeo.items())
        extrinsics = ioeo[3][1]
        intrinsics = ioeo[5][1]
        return intrinsics, extrinsics

    else:
        ioeo_list = pull_files(f'./ioeo/{camera}')
        target_date = datetime(int(year), int(month), int(day))
        closest_date = min(ioeo_list, key=lambda date: abs(date - target_date))
        ioeo = scipy.io.loadmat(f'./ioeo/{camera}/{closest_date.year}{closest_date.month:02d}{closest_date.day:02d}_{camera}_IOEOInitial.mat')
        ioeo = list(ioeo.items())
        extrinsics = ioeo[3][1]
        intrinsics = ioeo[5][1]
        return intrinsics, extrinsics


def processGCP(year, month, day, camera):

    gcp_file = pull_txt(year, month, day)  # pull text file of gcp data
    intrinsics, extrinsics = pull_ioeo(year, month, day, camera) # pull intrinsics and extrinsics

    #read_gcp file and output txt of points
    gcp_txt = pd.read_csv(gcp_file, delimiter=',',header=None).to_numpy()  # nx4 shape. index, easting, northing, ortho height
    ind, gcp_xyz = gcp_txt[:, 0], gcp_txt[:, 1:]  #split gcp index and xyz values

    # Calculate distance from camera to each GCP point
    gcp_dis = []
    for i in range(len(gcp_xyz) - 1): #excludes pier point gcp
        dis = cal_gcpDis(gcp_xyz[i], extrinsics, ind[i])
        gcp_dis.append([ind[i], dis])

    #convert xyz to uv
    UVd = xyz2DistUV(intrinsics[0], extrinsics[0], gcp_xyz[:-1])
    #round Ud and Vd to nearest pixel
    UV = np.round(UVd).astype(int)

    #remove points that are equal to zero in UV and gcp_dis
    gcp_dis = [gcp_dis[i] for i in range(len(gcp_dis)) if UV[0][i] != 0 and UV[1][i] != 0]
    UV = UV[:, (UV[0, :] != 0) | (UV[1, :] != 0)]

    return UV, ind, gcp_dis


def processVideo(file_path, create_frames=False, new_fps=30):

    capture = cv2.VideoCapture(file_path)

    original_fps = capture.get(cv2.CAP_PROP_FPS)  # get original fps of video

    if create_frames:

        new_fps = original_fps

        out_name = os.path.basename(file_path).split('.')[0] #create output name for frames
        # create output directory if it doesn't exist
        if not os.path.exists(f'./tower_images/video/{out_name}'):
            os.makedirs(f'./tower_images/video/{out_name}')

        frame_count = 0
        while capture.isOpened():

            ret, frame = capture.read()

            if not ret:
                print("Can't recieve frame. Exiting...")
                break

            frame_interval = int(original_fps/new_fps)  #define frame interval to create new fps

            if frame_count % frame_interval == 0:  #if the frame count is a multiple of the frame interval, save the frame

                cv2.imwrite(f'./tower_images/video/{out_name}/{out_name}' + '_' + str(frame_count) + '.jpeg', frame)  # save frame as JPEG file

            frame_count +=1

    return original_fps


def cal_rmse(estimated_depths, calculated_depths):

    return np.sqrt(np.sum([(calculated_depths[i] - estimated_depths[i]) ** 2 for i in range(len(calculated_depths))]) / len(calculated_depths))


def create_gif(plot_dir, temp_dir, name, image_files_prefix, fps=2):

    # gif_filename = f"{year}{month}{day}_{camera}_{name}.gif"
    gif_filename = f'{name}.gif'
    gif_path = os.path.join(plot_dir, gif_filename)

    # Collect and sort image files
    images = sorted(glob.glob(os.path.join(temp_dir, image_files_prefix + '*')), key=numerical_sort)
    imgs = [Image.open(img_file) for img_file in images]

    # Save GIF
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    print(f"GIF saved to {gif_path}")

    # Save the last frame as PNG
    # last_frame_filename = f"{year}{month}{day}_{camera}_{name}_last_frame.png"
    # last_frame_filename = f'{name}_last_frame.png'
    # last_frame_path = os.path.join(plot_dir, last_frame_filename)
    # imgs[-1].save(last_frame_path)
    # print(f"Last frame saved as PNG to {last_frame_path}")

    # Clean out temp directory
    for img_file in images:
        os.remove(img_file)


def create_input(file_path:str, flag:str='video', date:datetime=None, fps=1):

    if flag=='jaiabot':
        # pulls 6.1 jaiabot collect images from the LaCie to run on different models
        # set date
        year = str(date.year)
        month = f'{date.month:02d}'
        day = f'{date.day:02d}'

        imgs = glob.glob(f'{file_path}/{year}{month}{day}*/**/*.tiff', recursive=True) #pulls all images

        img_dict = defaultdict(list)  # sort images in dictionary by parent directory
        for img in imgs :
            parent_dir = os.path.dirname(img)
            img_dict[parent_dir].append(img)

        images = [sorted(imgs, key=numerical_sort)[:31] for imgs in img_dict.values() if imgs and len(sorted(imgs)) >= 31]  # grab the first n images for comparison from each camera

        return images, date

    if flag=='video':

        #pulls images from videos
        print(f'Processing video: {file_path}')
        original_fps = processVideo(file_path, False)
        img_paths = sorted(glob.glob(f'./tower_images/video/{os.path.basename(file_path).split(".")[0]}/*.jpeg'),key=numerical_sort)  # sort images by numerical order
        # cut images to desired fps
        n = round(original_fps / fps)
        img_paths = img_paths[::n]  # take every nth frame based on desired fps

        return img_paths


def run_dep_any(model, input:list, fps, gcp=False, date:datetime=None):

        dep_maps = [model.infer_image(cv2.imread(img)) for img in input if os.path.isfile(img)]

        depth_avg = gaussian_filter(dep_maps, sigma=3.0)
        dt = 1 / fps  # set dt for velocity calculation
        velocities = np.diff(dep_maps, axis=0) / dt
        stdDev = 2 * np.std(velocities, axis=0)  # twice the STD for 95% of data

        mean_depths = np.mean(dep_maps, axis=0)
        mean_velocity = np.mean(velocities, axis=0)
        output_path = './Depth_Anything_V2/temp/'

        estimated_depths = []
        calculated_depths = []

        for i in range(len(velocities)):
            raw_img = cv2.cvtColor(cv2.imread(input[i]), cv2.COLOR_BGR2RGB)

            if gcp:
                year = str(date.year)
                month = f'{date.month:02d}'
                day = f'{date.day:02d}'

                cam = re.split('_', os.path.basename(input[i]))  # take first image and split string
                camera = re.split('Cam', cam[0])[1] + cam[1]  # creates string Ex: 'BobA'

                UV, ind, gcp_dis = processGCP(year, month, day, camera)  # process GCP data

                est_dep = dep_maps[i][UV[1], UV[0]] #get estimated UV depth
                est_vel = velocities[i][UV[1], UV[0]]  # get estimated UV velocity

                # depth_plots.four_panel_gcp_velocity(raw_img, depth_avg[i], velocities[i], stdDev, UV, ind, est_vel, input[i], output_path)
                # depth_plots.four_panel_plot(raw_img, depth_avg[i], velocities[i], stdDev, input[i], output_path)

                # mean_est_depth = mean_depths[UV[1], UV[0]]  # get mean estimated depth
                # mean_est_vel = mean_velocity[UV[1], UV[0]]  # get mean estimated velocity

                if i == 0:
                    depth_plots.error_plot(est_dep, gcp_dis, cal_rmse(est_dep, np.array(gcp_dis)[:,1]), date, camera, output_path)  # create error plot for GCP depths
                    estimated_depths.extend(est_dep)
                    calculated_depths.extend(gcp_dis)  # append estimated and calculated depths to lists for error plot

            else:
                depth_plots.four_panel_plot(raw_img, dep_maps[i], velocities[i], stdDev, input[i], output_path)

        # create gif
        plot_dir = './Depth_Anything_V2/outputs/gif'
        temp_dir = './Depth_Anything_V2/temp'
        create_gif(plot_dir, temp_dir, name=f'{os.path.basename(os.path.dirname(input[0]))}_4hz',image_files_prefix='*', fps=fps)

        return estimated_depths, calculated_depths


def run_glpn(model, device, input:list, fps:int, gcp=False, date:datetime=None):

    # apply log transform to input
    images = [cv2.imread(img) for img in input if os.path.isfile(img)]
    cs = [255 / np.log(1 + np.max(img)) for img in images] # log transform constant
    log_img = [c * np.log(1 + img.astype(np.float32)) for c, img in zip(cs,images)]  # apply log transform to each image
    log_img = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) for img in log_img]  # normalize to 0-255

    # raw_img = np.array([cv2.resize(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), (640, 480)) for img in input if os.path.isfile(img)]) #resize for model input and convert images to RGB
    raw_img = np.array([cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 480)) for img in log_img]) #resize for model input and convert images to RGB
    original_shape = cv2.imread(input[0]).shape

    input_RGB = torch.tensor(raw_img, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Convert to tensor and permute to Batch x Channel x Height x Width
    input_RGB = input_RGB/255.0 #normalize input 0-1
    # input_RGB = input_RGB.unsqueeze(0)  # Add batch dimension if needed

    with torch.no_grad():
        pred = model(input_RGB)
    pred_d = pred['pred_d']

    #bring prediction back to the cpu
    pred_d = pred_d.squeeze().cpu().numpy() #remove batch dimension and bring to cpu
    #resize to original raw_img shape
    dep_maps = [cv2.resize(pred, (original_shape[1], original_shape[0])) for pred in pred_d]  # Resize to match original image size
    # est_dep = np.transpose(est_dep, (0, 1, 2)) # transpose if needed

    depth_avg = gaussian_filter(dep_maps, sigma=3.0)
    dt = 1 / fps  # set dt for velocity calculation
    velocities = np.diff(depth_avg, axis=0) / dt
    stdDev = 2 * np.std(velocities, axis=0)  # twice the STD for 95% of data

    mean_depths = np.mean(dep_maps, axis=0)
    mean_velocity = np.mean(velocities, axis=0)
    output_path = './GLPDepth/temp/'

    estimated_depths = []
    calculated_depths = []

    for i in range(len(velocities)):
        # raw_img = cv2.cvtColor(cv2.imread(input[i]), cv2.COLOR_BGR2RGB)
        raw_img = log_img[i]  # use log transformed image

        if gcp:
            year = str(date.year)
            month = f'{date.month:02d}'
            day = f'{date.day:02d}'

            cam = re.split('_', os.path.basename(input[i]))  # take first image and split string
            camera = re.split('Cam', cam[0])[1] + cam[1]  # creates string Ex: 'BobA'

            UV, ind, gcp_dis = processGCP(year, month, day, camera)  # process GCP data
            est_dep = dep_maps[i][UV[1], UV[0]]  # get estimated UV depth
            est_vel = velocities[i][UV[1], UV[0]]  # get estimated UV velocity

            # depth_plots.four_panel_gcp_velocity(raw_img, depth_avg[i], velocities[i], stdDev, UV, ind, est_vel, input[i], output_path)
            # depth_plots.four_panel_plot(raw_img, depth_avg[i], velocities[i], stdDev, input[i], output_path)

            # mean_est_depth = mean_depths[UV[1], UV[0]]  # get mean estimated depth
            # mean_est_vel = mean_velocity[UV[1], UV[0]]  # get mean estimated velocity

            if i == 0:
                depth_plots.error_plot(est_dep, gcp_dis, cal_rmse(est_dep, np.array(gcp_dis)[:,1]), date, camera,
                                       output_path)  # create error plot for GCP depths
                estimated_depths.extend(est_dep)
                calculated_depths.extend(gcp_dis)  # append estimated and calculated depths to lists for error plot

        else:
            depth_plots.four_panel_plot(raw_img, depth_avg[i], velocities[i], stdDev, input[i], output_path)

    # create gif
    plot_dir = './GLPDepth/outputs/gif'
    temp_dir = './GLPDepth/temp'
    create_gif(plot_dir, temp_dir, name=f'{os.path.basename(os.path.dirname(input[0]))}_2hz_logir',image_files_prefix='*', fps=fps)

    return estimated_depths, calculated_depths


def run_dpt_zoe(model, input:list, fps:int, gcp=False, date:datetime=None):

    dep_maps = [model.infer_pil(Image.open(img).convert('RGB')) for img in input if os.path.isfile(img)]  # load images and convert to RGB

    depth_avg = gaussian_filter(dep_maps, sigma=3.0)
    dt = 1 / fps  # set dt for velocity calculation
    velocities = np.diff(depth_avg, axis=0) / dt
    stdDev = 2 * np.std(velocities, axis=0)  # twice the STD for 95% of data

    mean_depths = np.mean(dep_maps, axis=0)
    mean_velocity = np.mean(velocities, axis=0)
    output_path = './ZoeDepth/temp/'

    estimated_depths = []
    calculated_depths = []

    for i in range(len(velocities)):
        raw_img = cv2.cvtColor(cv2.imread(input[i]), cv2.COLOR_BGR2RGB)

        if gcp:
            year = str(date.year)
            month = f'{date.month:02d}'
            day = f'{date.day:02d}'

            cam = re.split('_', os.path.basename(input[i]))  # take first image and split string
            camera = re.split('Cam', cam[0])[1] + cam[1]  # creates string Ex: 'BobA'

            UV, ind, gcp_dis = processGCP(year, month, day, camera)  # process GCP data
            est_dep = dep_maps[i][UV[1], UV[0]]  # get estimated UV depth
            est_vel = velocities[i][UV[1], UV[0]]  # get estimated UV velocity

            # depth_plots.four_panel_gcp_velocity(raw_img, depth_avg[i], velocities[i], stdDev, UV, ind, est_vel, input[i], output_path)
            # depth_plots.four_panel_plot(raw_img, depth_avg[i], velocities[i], stdDev, input[i], output_path)

            # mean_est_depth = mean_depths[UV[1], UV[0]]  # get mean estimated depth
            # mean_est_vel = mean_velocity[UV[1], UV[0]]  # get mean estimated velocity

            if i == 0:
                depth_plots.error_plot(est_dep, gcp_dis, cal_rmse(est_dep, np.array(gcp_dis)[:, 1]), date, camera,
                                       output_path)  # create error plot for GCP depths
                estimated_depths.extend(est_dep)
                calculated_depths.extend(gcp_dis)  # append estimated and calculated depths to lists for error plot

        else:
            depth_plots.four_panel_plot(raw_img, depth_avg[i], velocities[i], stdDev, input[i], output_path)

    # create gif
    # plot_dir = './ZoeDepth/outputs/gif'
    # temp_dir = './ZoeDepth/temp'
    # create_gif(plot_dir, temp_dir, name=f'{os.path.basename(os.path.dirname(input[0]))}_2hz_dis', image_files_prefix='*beach*', fps=fps)

    return estimated_depths, calculated_depths


def run_dep_pro(model, transform, input:list, fps:int, gcp=False, date:datetime=None):

    prediction = [model.infer(transform(depth_pro.load_rgb(img)[0])) for img in input if os.path.isfile(img)]  # load images and convert to RGB

    dep_maps = [pred['depth'].cpu().numpy() for pred in prediction] #get depth outputs and bring to cpu

    depth_avg = gaussian_filter(dep_maps, sigma=3.0)
    dt = 1 / fps  # set dt for velocity calculation
    velocities = np.diff(depth_avg, axis=0) / dt
    stdDev = 2 * np.std(velocities, axis=0)  # twice the STD for 95% of data

    mean_depths = np.mean(dep_maps, axis=0)
    mean_velocity = np.mean(velocities, axis=0)
    output_path = './ml_depth_pro/temp/'

    estimated_depths = []
    calculated_depths = []

    for i in range(len(velocities)):
        raw_img = cv2.cvtColor(cv2.imread(input[i]), cv2.COLOR_BGR2RGB)

        if gcp:
            year = str(date.year)
            month = f'{date.month:02d}'
            day = f'{date.day:02d}'

            cam = re.split('_', os.path.basename(input[i]))  # take first image and split string
            camera = re.split('Cam', cam[0])[1] + cam[1]  # creates string Ex: 'BobA'

            UV, ind, gcp_dis = processGCP(year, month, day, camera)  # process GCP data
            est_dep = dep_maps[i][UV[1], UV[0]]  # get estimated UV depth
            est_vel = velocities[i][UV[1], UV[0]]  # get estimated UV velocity

            # depth_plots.four_panel_gcp_velocity(raw_img, depth_avg[i], velocities[i], stdDev, UV, ind, est_vel, input[i], output_path)
            depth_plots.four_panel_plot(raw_img, depth_avg[i], velocities[i], stdDev, input[i], output_path)

            # mean_est_depth = mean_depths[UV[1], UV[0]]  # get mean estimated depth
            # mean_est_vel = mean_velocity[UV[1], UV[0]]  # get mean estimated velocity

            if i == 0:
                depth_plots.error_plot(est_dep, gcp_dis, cal_rmse(est_dep, np.array(gcp_dis)[:,1]), date, camera,
                                       output_path)  # create error plot for GCP depths
                estimated_depths.extend(est_dep)
                calculated_depths.extend(gcp_dis)  # append estimated and calculated depths to lists for error plot

        else:
            depth_plots.four_panel_plot(raw_img, depth_avg[i], velocities[i], stdDev, input[i], output_path)

    # create gif
    # plot_dir = './ml_depth_pro/outputs/gif'
    # temp_dir = './ml_depth_pro/temp'
    # create_gif(plot_dir, temp_dir, name=f'{os.path.basename(os.path.dirname(input[0]))}_2hz_dis',image_files_prefix='*beach*', fps=fps)

    return estimated_depths, calculated_depths


def run_towerframes(model, input_path, device:str, run_model:str, gcp:bool=True):

    bob_estDeps = []  # bob cams only
    bob_calDeps = []
    mary_estDeps = []  # mary cams only
    mary_calDeps = []

    date_list = pull_files(input_path)
    for date in tqdm(date_list):
        inputs, day = create_input(input_path, 'jaiabot', date)
        global est_dep, cal_dep

        for input in inputs:
            # run model - get prediction
            if run_model == 'dep_any':
                est_dep, cal_dep = run_dep_any(model, input, 2, gcp=gcp, date=day)  #TODO: save a numpy file of the estimated dep map in the data folder
            elif run_model == 'glpn':
                est_dep, cal_dep = run_glpn(model, device, input, 2, gcp=gcp, date=day)
            elif run_model == 'dpt_zoe':
                est_dep, cal_dep = run_dpt_zoe(model, input, 2, gcp=gcp, date=day)
            elif run_model == 'dep_pro':
                est_dep, cal_dep = run_dep_pro(model, transform, input, 2, gcp=gcp, date=day)

            if gcp==True:

                cam = re.split('_', os.path.basename(input[0]))  # take first image and split string
                camera = re.split('Cam', cam[0])[1] + cam[1]  # creates string Ex: 'BobA'

                # append estimated depths and calculated depths to lists
                if camera.startswith('Bob'):
                    bob_estDeps.extend(est_dep)
                    bob_calDeps.extend([cal_dep[i][1] for i in range(len(cal_dep))])

                else:
                    mary_estDeps.extend(est_dep)
                    mary_calDeps.extend([cal_dep[i][1] for i in range(len(cal_dep))])

    #save all lists
    np.savez('./data/dep_pro_gcp.npz',
             bob_estDeps=bob_estDeps,
             bob_calDeps=bob_calDeps,
             mary_estDeps=mary_estDeps,
             mary_calDeps=mary_calDeps)


def run_video(model, input_path, device:str, run_model:str, fps:int=4):

    for file in input_path[:]:

        inputs = create_input(file, 'video', fps=fps)

        if run_model == 'dep_any':
            run_dep_any(model, inputs, fps)
        elif run_model == 'glpn':
            run_glpn(model, device, inputs, fps)
        elif run_model == 'dpt_zoe':
            run_dpt_zoe(model, inputs, fps)
        elif run_model == 'dep_pro':
            run_dep_pro(model, transform, inputs, fps)


if __name__ == '__main__':

    #select device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    #select model
    # model = models.dep_any(device, pred='metric')
    model = models.glpn(device)
    # model = models.intel_zoe(device)
    # model, transform = models.dep_pro(device)

    #IR data
    input_path = ['/mnt/e/surrogate_lwir_data/skyraiderR80D/fov_offshore/20250508F01_SRH701384881_IR_0007_reverseTransit.TS']
    # processVideo(input_path[0], create_frames=True)

    '''RUN VIDEO'''
    # input_path = glob.glob('./tower_images/video/*.MOV')
    run_video(model, input_path, device, run_model='glpn', fps=2)

    '''RUN TOWERFRAMES'''
    # input_path = '/mnt/e/towerframes'
    # run_towerframes(model, input_path, device, run_model='dep_pro', gcp=True)

    # dep_any_data = np.load('./data/depAny_gcp.npz')
    # bob_estDeps = dep_any_data['bob_estDeps']
    # bob_calDeps = dep_any_data['bob_calDeps']
    # mary_estDeps = dep_any_data['mary_estDeps']
    # mary_calDeps = dep_any_data['mary_calDeps']
    #
    # output_path = './Depth_Anything_V2/temp/'
    # depth_plots.error_comparison(
    #     [bob_estDeps, mary_estDeps],
    #     [bob_calDeps, mary_calDeps],
    #     ['Bob', 'Mary'],
    #     [cal_rmse(bob_estDeps, bob_calDeps), cal_rmse(mary_estDeps, mary_calDeps)],
    #     output_path, vmax=60)  # create error comparison plot for all cameras



    # img = cv2.imread('./tower_images/video/S1031511_IR_forwardTransit/S1031511_IR_forwardTransit_413.jpeg')
    # c = 255 / np.log(1 + np.max(img))  #log transform constant
    # log_img = c * np.log(1 + img.astype(np.float32))
    # log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # normalize to 0-255
    #
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # ax.imshow(log_img)
    # ax.set_title('log_img')
    # ax2.imshow(img)
    # ax2.set_title('original_img')
    # ax.axis('off')  # Hide the axes
    # ax2.axis('off')
    # plt.tight_layout()
    # plt.show()
    # plt.close('all')

