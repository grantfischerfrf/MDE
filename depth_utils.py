from image_processing import xyz2DistUV
from ml_depth_pro.src import depth_pro

import re
import os
import cv2
import glob
import torch
import pandas as pd
import numpy as np
import scipy
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict


def numerical_sort(string):

    parts = re.split(r'(\d+)', string)
    return [int(part) if part.isdigit() else part for part in parts]


def pull_dictionary_data(file_path:str):

    try:
        dict={}
        data = np.load(file_path, allow_pickle=True)
        name = os.path.basename(os.path.dirname(file_path))
        dict['name'] = name
        dict['messages'] = data['messages']
        dict['timestamps'] = data['timestamps'] / 1000000000  # Convert nanoseconds to seconds
        return dict

    except Exception as e:
        print(f'Error loading file {file_path}: {e}')
        print('If bad zip file, try manually extracting first')


def find_reference_dict(dicts:list):

    #finds the dictionary with the smallest timestamp difference
    best_dict=None
    best_diff=float('inf')
    for d in dicts:
        timestamps= d['timestamps']
        dt = timestamps[-1] - timestamps[0]
        if dt < best_diff:
            best_diff = dt
            best_dict = d

    return best_dict


def compare_timestamps(dicts:list):
    """
        Aligns data from multiple dictionaries by matching their timestamps to a reference timestamp.

        This function selects a reference dictionary (using `find_reference_dict`),
        then aligns the 'messages' and 'timestamps' in the remaining dictionaries
        to the reference timestamps by finding the closest matching timestamp index.
        Returns a dictionary where each entry is sorted to match the reference.

        Args:
            dicts (list): A list of dictionaries. Each dictionary must contain the following keys:
                - 'name' (str): A name/unique identifier.
                - 'timestamps' (np.ndarray): Array of timestamps.
                - 'messages' (np.ndarray): Raw image array corresponding to the timestamps.

        Returns:
            dict: A dictionary keyed by the 'name' field of each input dictionary.
                  Each value is a dictionary with aligned 'messages' and 'timestamps'.
        """
    #Compares timestamps and finds the closest match
    ref_dict = find_reference_dict(dicts)
    ref_timestamps = ref_dict['timestamps']

    #loop through the rest of the dictionaries and sort data by timestamps
    sorted_dict = {}
    for d in dicts:
        if d is not ref_dict:
            d_timestamps = d['timestamps']
            d_data = d['messages']

            #find the closest match for each timestamp in ref_dict
            closest_indices = []
            for ref_time in ref_timestamps:
                closest_index = np.argmin(np.abs(d_timestamps - ref_time))
                closest_indices.append(closest_index)

            #sort the data by the closest indices
            d_sorted_data = d_data[closest_indices]
            d_sorted_time = d_timestamps[closest_indices]

            d['messages'] = d_sorted_data
            d['timestamps'] = d_sorted_time
            sorted_dict[f'{d['name']}'] = d

        else:
            sorted_dict[f'{d['name']}'] = d

    return sorted_dict


def read_image(input):

    if type(input) == str:
        # uses cv2 to read raw image from file path
        return cv2.imread(input)

    #if image is already raw array
    elif input.ndim == 3:
        return input

    #if image array has no channels expand dimensions
    else:
        input = np.expand_dims(input, axis=-1)  # Add channel dimension
        return input


def save_data(save_path: str, data: np.ndarray):

    # Pickles and saves output depth mask from MDE model
    np.save(save_path, data, allow_pickle=True)


def calculate_gcpDistance(gcp_xyz, extrinsics, ind):

    # Calculate distance from camera to GCP point
    gcp_dis = np.sqrt((gcp_xyz[0] - extrinsics[0][0]) ** 2 + (gcp_xyz[1] - extrinsics[0][1]) ** 2 + (gcp_xyz[2] - extrinsics[0][2]) ** 2)
    # print(f'GCP-{ind} distance from camera: {gcp_dis}m')
    return gcp_dis


def pull_file_dates(file_path):

    # use datetime to pull all data collect dates from folder
    date_list = os.listdir(file_path)
    date_list = [datetime.strptime(match.group(), '%Y%m%d') for date in date_list if (match := re.match(r'^\d{8}', date))]

    return date_list


def pull_GCP_txt_file(year, month, day):

    if os.path.isfile(f'./txt/{year}{month}{day}'):
        return f'./txt/{year}{month}{day}.txt'
    else:
        txt_list = pull_file_dates('./txt')
        target_date = datetime(int(year), int(month), int(day))
        closest_date = min(txt_list, key=lambda date: abs(date - target_date))
        return f'./txt/{closest_date.year}{closest_date.month:02d}{closest_date.day:02d}.txt'


def pull_GCP_ioeo_file(year, month, day, camera):

    if os.path.isfile(f'./ioeo/{camera}/{year}{month}{day}_{camera}_IOEOInitial.mat'):
        ioeo = scipy.io.loadmat(f'./ioeo/{camera}/{str(year) + str(month) + str(day)}_{camera}_IOEOInitial.mat')
        ioeo = list(ioeo.items())
        extrinsics = ioeo[3][1]
        intrinsics = ioeo[5][1]
        return intrinsics, extrinsics

    else:
        ioeo_list = pull_file_dates(f'./ioeo/{camera}')
        target_date = datetime(int(year), int(month), int(day))
        closest_date = min(ioeo_list, key=lambda date: abs(date - target_date))
        ioeo = scipy.io.loadmat(f'./ioeo/{camera}/{closest_date.year}{closest_date.month:02d}{closest_date.day:02d}_{camera}_IOEOInitial.mat')
        ioeo = list(ioeo.items())
        extrinsics = ioeo[3][1]
        intrinsics = ioeo[5][1]
        return intrinsics, extrinsics


def process_GCP_points(year:str, month:str, day:str, camera:str):
    """
    Processes Ground Control Points (GCPs) for a specific date and camera.

    This function performs the following steps:
    - Loads the GCP coordinate text file for the given date.
    - Loads camera intrinsics and extrinsics (ioeo file) for the given date.
    - Computes distances from the camera to each GCP point (excluding the last one - data point for another study).
    - Projects 3D GCP points into 2D image coordinates (UV space).
    - Filters out UV coordinates that project to outside the image bounds -> set to (0, 0).

    Args:
        year (str): Year of the GCP data collection.
        month (str): Month of the GCP data collection.
        day (str): Day of the GCP data collection.
        camera (str): Camera identifier for which IOEO data is pulled.

    Returns:
        tuple:
            - UV (np.ndarray): 2D array of pixel coordinates (2 x N) after projection and filtering.
            - ind (np.ndarray): 1D array of GCP numbers.
            - gcp_dis (list): List of [index, distance] pairs for GCPs including the GCP# and distance from the camera in meters.
    """

    gcp_file = pull_GCP_txt_file(year, month, day)  # pull text file of gcp data
    intrinsics, extrinsics = pull_GCP_ioeo_file(year, month, day, camera) # pull intrinsics and extrinsics

    #read_gcp file and output txt of points
    gcp_txt = pd.read_csv(gcp_file, delimiter=',',header=None).to_numpy()  # nx4 shape. index, easting, northing, ortho height
    ind, gcp_xyz = gcp_txt[:, 0], gcp_txt[:, 1:]  #split gcp index and xyz values

    # Calculate distance from camera to each GCP point
    gcp_dis = []
    for i in range(len(gcp_xyz) - 1): #excludes pier point gcp
        dis = calculate_gcpDistance(gcp_xyz[i], extrinsics, ind[i])
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

        # new_fps = original_fps

        out_name = os.path.basename(file_path).split('.')[0] #create output name for frames
        # create output directory if it doesn't exist
        if not os.path.exists(f'./tower_images/video/{out_name}'):
            os.makedirs(f'./tower_images/video/{out_name}') #FIXME: change this and the cv2.imwrite to a generic path

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


def calculate_rmse(estimated_depths, calculated_depths):

    return np.sqrt(np.sum([(calculated_depths[i] - estimated_depths[i]) ** 2 for i in range(len(calculated_depths))]) / len(calculated_depths))


def create_input_for_MDE(file_path:str, flag:str='video', date:datetime=None, fps=1):
    """
     Creates a list of image file paths to be used as input for an MDE model.

     This function supports two modes of operation:
     - `'jaiabot'`: Gathers .tiff images collected for the 6.1 hazardous hydro project on a specific date.
     - `'video'`: Extracts frames from a video file at a specified frame-per-second (fps) rate.

     Args:
         file_path (str):
             - For `'jaiabot'`: Root directory containing image folders named by date: YYYYMMDD*.
             - For `'video'`: Path to the input video file.
         flag (str, optional):
             Specifies the mode of operation: `'jaiabot'` or `'video'`. Defaults to `'video'`.
         date (datetime, optional):
             Required only if `flag` is `'jaiabot'`. Specifies the collection date of the images.
         fps (int, optional):
             Desired frame extraction rate for videos (frames per second). Defaults to 1.

     Returns:
         tuple or list:
             - If `flag` is `'jaiabot'`: Returns a tuple (`images`, `date`) where:
                 - `images` (list of list): Each sublist contains 'n' sorted `.tiff` image paths from a camera directory.
                 - `date` (datetime): The input date for reference.
             - If `flag` is `'video'`: Returns a list of `.jpeg` image paths sampled at the specified `fps`.

     Raises:
         ValueError: If `flag` is `'jaiabot'` but `date` is not provided.
     """

    if flag=='jaiabot':
        # pulls 6.1 jaiabot collect images to run on different models
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


def run_dep_any(model, images, input=None, data_name:str=None):

    #run inference using Depth Anything V2 model
    dep_maps = [model.infer_image(read_image(img)) for img in images[:]]

    if type(input) == dict:
        input['messages'] = np.array(dep_maps)
        input['name'] = str(input['name']) + '_mde'
        save_data(f'./Depth_Anything_V2/dep_any_data/{data_name}_mde', input)
        print('Data saved.')

    else:
        input = np.array(dep_maps)
        save_data(f'./Depth_Anything_V2/dep_any_data/{data_name}_mde', input)
        print('Data saved.')  #TODO: add camera name for towerframes at some point if needed. Overwrites if subfolders exist inside the data folder.

    return dep_maps


def run_glpn(model, device, input, data_name:str=None):

    #run inference using GLPDepth model
    raw_img = np.array([cv2.resize(cv2.cvtColor(read_image(img), cv2.COLOR_BGR2RGB), (640, 480)) for img in input]) #resize for model input and convert images to RGB
    original_shape = read_image(input[0]).shape

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

    if type(input) == dict:
        input['messages'] = np.array(dep_maps)
        input['name'] = str(input['name']) + '_mde'
        save_data(f'./GLPDepth/glpn_data/{data_name}_mde', input)
        print('Data saved.')

    else:
        input = np.array(dep_maps)
        save_data(f'./GLPDepth/glpn_data/{data_name}_mde', input)
        print('Data saved.')


def run_dpt_zoe(model, input, data_name:str=None):

    #run inference using ZoeDepth model
    if type(input) == dict:
        dep_maps = [model.infer_pil(Image.fromarray(img).convert('RGB')) for img in input]

        input['messages'] = np.array(dep_maps)
        input['name'] = str(input['name']) + '_mde'
        save_data(f'./ZoeDepth/zoe_dep_data/{data_name}_mde', input)
        print('Data saved.')

    else:
        dep_maps = [model.infer_pil(Image.open(img).convert('RGB')) for img in input if os.path.isfile(img)]  # load images and convert to RGB

        input = np.array(dep_maps)
        save_data(f'./ZoeDepth/zoe_dep_data/{data_name}_mde', input)
        print('Data saved.')


def run_dep_pro(model, transform, input, data_name:str=None):

    #run inference using Depth Pro model
    if type(input) == dict:
        prediction = [model.infer(transform(Image.fromarray(img))) for img in input]
        dep_maps = [pred['depth'].cpu().numpy() for pred in prediction]  # get depth outputs and bring to cpu

        input['messages'] = np.array(dep_maps)
        input['name'] = str(input['name']) + '_mde'
        save_data(f'./ml_depth_pro/dep_pro_data/{data_name}_mde', input)
        print('Data saved.')

    else:
        prediction = [model.infer(transform(depth_pro.load_rgb(img)[0])) for img in input if os.path.isfile(img)]  # load images and convert to RGB
        dep_maps = [pred['depth'].cpu().numpy() for pred in prediction] #get depth outputs and bring to cpu

        input = np.array(dep_maps)
        save_data(f'./ml_depth_pro/dep_pro_data/{data_name}_mde', input)
        print('Data saved.')


def find_GCP_depths(dep_maps, date:datetime, input, fps:int=1):
    #calculated depth at GCP points
    dt = 1 / fps  # set dt for velocity calculation
    velocities = np.diff(dep_maps, axis=0) / dt
    stdDev = 2 * np.std(velocities, axis=0)  # twice the STD for 95% of data

    year = str(date.year)
    month = f'{date.month:02d}'
    day = f'{date.day:02d}'

    cam = re.split('_', os.path.basename(input[0]))  # take first image and split string
    camera = re.split('Cam', cam[0])[1] + cam[1]  # creates string Ex: 'BobA'

    UV, ind, cal_dep = process_GCP_points(year, month, day, camera)  # process GCP data

    estimated_depths = []
    calculated_depths = []

    for i in range(len(velocities)):
        est_dep = dep_maps[i][UV[1], UV[0]]  # get estimated UV depth
        est_vel = velocities[i][UV[1], UV[0]]  # get estimated UV velocity - variable not used, here if needed in the future

        if i == 0:
            estimated_depths.extend(est_dep)
            calculated_depths.extend(cal_dep)

    return estimated_depths, calculated_depths, camera


def run_towerframes(model, input_path, device:str, run_model:str, data_name:list, gcp:bool=False, transform=None):
    """
    Runs a depth estimation model on tower frame images (6.1 hazardous hydro. project data)),
    optionally comparing predictions to Ground Control Point (GCP) depth values.

    This function processes frames for each date:
    - Loads tower images using `create_input_for_MDE` with the 'jaiabot' flag.
    - Runs the specified model to generate depth maps.
    - If `gcp` is True, calculates estimated depths from predictions and compares them
      with calculated GCP-based depths using `find_GCP_depths`.
    - Organizes outputs into separate lists for Bob and Mary camera groups.
    - Saves all estimated and GCP-calculated depth values to a `.npz` file.

    Args:
        model:
            Preloaded depth estimation model.
        input_path (str):
            Path to the root folder containing input image data.
        device (str):
            Device identifier (`'cpu'` or `'cuda'`) for model execution.
        run_model (str):
            Specifies the model to use. Must be one of:
            `'dep_any'`, `'glpn'`, `'dpt_zoe'`, or `'dep_pro'`.
        data_name (list):
            List of dataset names or camera IDs corresponding to each date.
        gcp (bool, optional):
            Whether to compare model predictions against GCP depths. Defaults to False.
        transform (callable, optional):
            A transformation function applied to inputs for the `'dep_pro'` model.

    Returns:
        None

    Saves:
        Saves a `.npz` file to `./data/{run_model}_gcp.npz` containing:
            - `bob_estDeps`: Estimated depths from Bob cameras.
            - `bob_calDeps`: GCP-based depths for Bob cameras.
            - `mary_estDeps`: Estimated depths from Mary cameras.
            - `mary_calDeps`: GCP-based depths for Mary cameras.

    Raises:
        ValueError: If an unknown model name is passed to `run_model`.
    """

    bob_estDeps = []  # bob cams only
    bob_calDeps = []
    mary_estDeps = []  # mary cams only
    mary_calDeps = []

    date_list = pull_file_dates(input_path)
    for date, name in tqdm(zip(date_list, data_name), total=(len(date_list))):
        inputs, day = create_input_for_MDE(input_path, 'jaiabot', date)

        for input in inputs:
            # run model - get prediction
            if run_model == 'dep_any':
                dep_maps = run_dep_any(model, input, data_name=name)
            elif run_model == 'glpn':
                dep_maps = run_glpn(model, device, input, data_name=name)
            elif run_model == 'dpt_zoe':
                dep_maps = run_dpt_zoe(model, input, data_name=name)
            elif run_model == 'dep_pro':
                dep_maps = run_dep_pro(model, transform, input, data_name=name)
            else:
                raise ValueError(f'Unknown model type: {run_model}')

            if gcp:
                est_dep, cal_dep, camera = find_GCP_depths(dep_maps, date, input, fps=1)

                # append estimated depths and calculated depths to lists
                if camera.startswith('Bob'):
                    bob_estDeps.extend(est_dep)
                    bob_calDeps.extend([cal_dep[i][1] for i in range(len(cal_dep))])

                else:
                    mary_estDeps.extend(est_dep)
                    mary_calDeps.extend([cal_dep[i][1] for i in range(len(cal_dep))])

    #save all lists
    np.savez(f'./data/{run_model}_gcp.npz',
             bob_estDeps=bob_estDeps,
             bob_calDeps=bob_calDeps,
             mary_estDeps=mary_estDeps,
             mary_calDeps=mary_calDeps)


def run_video(model, input_path, device:str, run_model:str, data_name:list, fps:int=4, transform=None):
    """
    Runs a depth estimation model on a series of videos.

    For each video file:
    - Extracts frames at a specified frame-per-second (fps) rate using `create_input_for_MDE`.
    - Runs the specified model on the extracted frames using the appropriate inference function.

    Args:
        model:
            The preloaded model to use for inference.
        input_path (list):
            List of paths to input video files.
        device (str):
            Device to run the model on (`'cpu'` or `'cuda'`).
        run_model (str):
            Specifies which model to use. Must be one of:
            `'dep_any'`, `'glpn'`, `'dpt_zoe'`, or `'dep_pro'`.
        data_name (list):
            List of names corresponding to each video file. Used for labeling or saving output.
        fps (int, optional):
            Frame extraction rate from video in frames per second. Defaults to 4.
        transform (callable, optional):
            A transformation function to be applied to the input, only used for `'dep_pro'`.

    Returns:
        None

    Raises:
        ValueError: If `run_model` is not one of the supported model types.
    """

    for file, name in zip(input_path[:], data_name):

        inputs = create_input_for_MDE(file, 'video', fps=fps)

        if run_model == 'dep_any':
            run_dep_any(model, inputs, data_name=name)
        elif run_model == 'glpn':
            run_glpn(model, device, inputs, data_name=name)
        elif run_model == 'dpt_zoe':
            run_dpt_zoe(model, inputs, data_name=name)
        elif run_model == 'dep_pro':
            run_dep_pro(model, transform, inputs, data_name=name)


def run_rawImages(model, inputs:list, device:str, run_model:str, data_name:list, transform=None):
    """
    Runs a depth estimation model on a series of dictionaries containing raw image arrays.

    For each input path, this function:
    - Loads raw image data from a serialized dictionary.
    - Selects and runs the specified model variant (`dep_any`, `glpn`, `dpt_zoe`, or `dep_pro`).
    - Passes additional arguments (e.g., device, transform, and data_name) to model-specific runners.

    Args:
        model:
            The preloaded model to use for inference.
        inputs (list):
            list of paths to raw image dictionary files.
        device (str):
            Device to run the model on (`'cpu'` or `'cuda'`).
        run_model (str):
            Specifies which model to use. Must be one of:
            `'dep_any'`, `'glpn'`, `'dpt_zoe'`, or `'dep_pro'`.
        data_name (list):
            List of names corresponding to each input. Used for labeling or saving output.
        transform (callable, optional):
            A transformation function to be applied to the input, only used for `'dep_pro'`.

    Returns:
        None

    Raises:
        ValueError: If `run_model` is not one of the supported options.
    """
    #pull data
    for path, name in zip(inputs, data_name):
        input = pull_dictionary_data(path)
        raw_img = input['messages']

        #run model
        if run_model == 'dep_any':
            run_dep_any(model, raw_img, input, data_name=name)
        elif run_model == 'glpn':
            run_glpn(model, device, input, data_name=name)
        elif run_model == 'dpt_zoe':
            run_dpt_zoe(model, input, data_name=name)
        elif run_model == 'dep_pro':
            run_dep_pro(model, transform, input, data_name=name)

