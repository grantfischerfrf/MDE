# Monocular Depth Estimation Toolbox

This repository contains a collection of Monocular Depth Estimation (MDE) models 
for analysis on different datasets. Current functionalities of this repository include: 
* Comparing MDE to Stereo depth estimation (sample data)
* Running MDE model inference on raw image arrays
* Running MDE model inference on video
* Running MDE model inference on Jack-up tower images (FRF data)


**Install Dependencies:**
```
git clone https://github.com/grantfischerfrf/MDE.git
cd MDE
pip install -r requirements.txt
```

### Download Pretrained Weights

Instructions for downloading pretrained model weights can be found inside each model's respective repository.
These are included as submodules in this repository.

Place the weights in the 'checkpoints' folder inside the directory named for the model.

Example
: *For Depth_Anything_V2 - Metric*: Go to the Depth_Anything_V2 directory and download the desired weights. 
Then, place the downloaded weights in the 'checkpoints' folder inside the 
'Depth_Anything_V2/metric_depth/' directory.


### Download Sample Data

#### Example.ipynb provides a walkthrough of how to run the MDE models on the sample data and visualize the outputs

The sample data was collected using two stereo camera rigs mounted onto the LARC, 
an amphibious vehicle at the USACE-CHL Field Research Facility.
The first camera system consisted of two monochrome cameras supplemented by an RGB camera.
The second camera system consisted of two near-infrared (NIR) cameras.

Sample data can be downloaded from the following link:

https://drive.google.com/drive/folders/1f2W4gYgDx50dvqlsllMIsZnIWEsqbnOw?usp=sharing

#### Data Naming Convention
**ms_output_#**: identifies the specific data collection to which the output data belongs.

Data labeled **ms1** corresponds to the NIR cameras.
Data labeled **ms2** corresponds to the monochrome and RGB cameras.

After downloading the sample data, place each file in the appropriate directory in the 'data' folder. 
Each file is named according to the sub-folder structure it should be placed in. 

In addition, each file has a description of the data contained inside:

*left_depth*: Stereo depth map, aligned with the left image.

*left_image_rect*: Left stereo input, corrected for distortion

*right_image_rect*: Right stereo input, corrected for distortion

*aux_image_rect_color*: RGB image, corrected for distortion

### Scripts

**main.py**: Main access point for running MDE models and visualizing results. 

**models.py**: Contains the model classes for each MDE model. Configures the model and loads in weights.

**depth_utils.py** Contains support functions for executing model inference.

**depth_plot.py** Contains plotting functions to visualize results and compute error metrics.  

**plot_utils.py** Contains data processing functions to support visualizations.

**image_processing.py** Contains functions for reprojecting 3D GCP points into 2D image coordinates (UV space). 
This is an old script used to compare MDE predictions to ground truth data.
Originally used on the jack-up tower data from 6.1 hazardous hydrodynamics project.

### Saving

* MDE model outputs are saved in a data folder under the directory of the model used.
Ex: if you used Depth_Anything_V2, the outputs will be saved in 'Depth_Anything_V2/dep_any_data/'

* It is recommended to save visualizations in the output folder under the directory of the
respective model used for organization purposes.





