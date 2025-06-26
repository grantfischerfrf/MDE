import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.gridspec import GridSpec
import corefunctions as cf
# from coastalimagelib import supportfunctions as sf
# import jblib
import datetime as DT
import os

def extract_int_ext(year, month, day, camera):

    # save the original day in case there is no ioeo file
    original_day = day

    # Load the intrinsics and extrinsics from the IOEO file
    print(f'./ioeo/{camera}/{str(year) + str(month) + str(day)}_{camera}_IOEOInitial.mat')
    if os.path.isfile(f'./ioeo/{camera}/{str(year) + str(month) + str(day)}_{camera}_IOEOInitial.mat'):
        ioeo = scipy.io.loadmat(f'./ioeo/{camera}/{str(year) + str(month) + str(day)}_{camera}_IOEOInitial.mat')
        ioeo = list(ioeo.items())
        extrinsics = ioeo[3][1]
        intrinsics = ioeo[5][1]
        return intrinsics, extrinsics, year, month, day

    if not os.path.isfile(f'./ioeo/{camera}/{str(year) + str(month) + str(day)}_{camera}_IOEOInitial.mat'):
        print('IOEO file does not exist, using previously defined extrinsics')
        while not os.path.isfile(f'./ioeo/{camera}/{str(year) + str(month) + str(day)}_{camera}_IOEOInitial.mat'):
            month = int(month)
            day = int(day)
            if day == 0:
                month -= 1
                day = 31
            else:
                day -= 1
        ioeo = scipy.io.loadmat(f'./ioeo/{camera}/{str(year) + str(month) + str(day)}_{camera}_IOEOInitial.mat')
        ioeo = list(ioeo.items())
        extrinsics = ioeo[3][1]
        intrinsics = ioeo[5][1]

        #reverts the day back to its original value so the correct images will be pulled from rectification_local
        day = original_day

        return intrinsics, extrinsics, int(year), int(month), int(day)


def rectification_local(intrinsics, extrinsics, year, month, day, cameras):
    # Grid specifications
    xMin = 0
    xMax = 500
    yMin = 0
    yMax = 1200
    dy = 1
    dx = 1
    z = 0

    #FRF origin and local angle in NAD83 NCSP coordinates
    angle = 20.0253
    origin = [901951.6805, 274093.1562, np.deg2rad(angle)]

    grid = cf.XYZGrid([xMin, xMax], [yMin, yMax], dx, dy, z)

    new_cameras = np.empty(len(cameras), dtype=object)

    file_base = []

    for i in range(len(cameras)):
        folder = f'./gcp_images/{int(year[i])}/{int(month[i])}/{int(day[i])}/{cameras[i]}/'
        file = [os.path.join(folder, file) for file in os.listdir(folder)]
        file_base.append(file[0])
        print("Camera, intrinsics, extrinsics, file_base: ", cameras[i], intrinsics[i], extrinsics[i], file_base[i])
        new_cameras[i] = cf.CameraData(intrinsics[i], extrinsics[i], coords='geo', origin=origin, mType='CIRN', nc=3)
        new_cameras[i].addUV(grid)

    rect_frame = cf.mergeRectify(file_base, new_cameras, grid)

    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(rect_frame)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    if len(cameras) > 1:  # checks to see if multiple cameras are being used
        cameras = [''.join(cameras)]  # if so save the camera names as a single string

    if os.path.exists(f'./rectified_images/{year[i]}/{month[i]}/{day[i]}/{cameras[0]}') == False:  #checks to see if the rectified images already exist
        os.makedirs(f'./rectified_images/{year[i]}/{month[i]}/{day[i]}/{cameras[0]}')  #if not, create the directory

    plt.savefig(f'./rectified_images/{year[i]}/{month[i]}/{day[i]}/{cameras[0]}/{str(year[i])+str(month[i])+str(day[i])}_local_rect_'
            f'Cam{str(cameras) + str(file_base[0][-19:-6])}.png', dpi='figure', format='png', bbox_inches='tight', pad_inches=0.0)

    # plt.show()
    plt.close()

    return xMin, xMax, yMin, yMax, new_cameras, rect_frame


def xyz2DistUV(intrinsics, extrinsics, xyz):
    '''
    Converts XYZ world coordinates to UV image coordinates
    :param intrinsics: 1x11 array of camera intrinsics
    :param extrinsics: 1x6 array of camera extrinsics
    :param xyz: nx3 array of world coordinates

    :return:
        Ud: array of distorted U image coordinates
        Vd: array of distorted V image coordinates
    '''

    K, R, IC, P, C = intrinsicsExtrinsics2P(intrinsics, extrinsics)

    xyz_transpose = xyz.conj().transpose()
    ones = np.ones((1, xyz.shape[0]))
    xyz_aug = np.vstack((xyz_transpose, ones))
    UV = P @ xyz_aug

    UV = UV / np.tile(UV[2, :], (3, 1))

    U = UV[0, :]
    V = UV[1, :]

    Ud, Vd, flag = distortUV(U, V, intrinsics)

    # flag negative z values
    xyzC = R @ (IC @ xyz_aug)
    bind = np.where(xyzC[2, :] <= 0)
    flag[bind] = 0

    UVd = np.vstack((Ud, Vd))

    # return Ud * flag, Vd * flag
    return UVd


def intrinsicsExtrinsics2P(intrinsics, extrinsics):

    #create K matrix
    fx = intrinsics[4]
    fy = intrinsics[5]
    c0U = intrinsics[2]
    c0V = intrinsics[3]

    K = [[-fx, 0, c0U], [0, -fy, c0V], [0, 0, 1]]

    #create R matrix
    azimuth = extrinsics[3]
    tilt = extrinsics[4]
    swing = extrinsics[5]

    R = Angles2R(azimuth, tilt, swing)

    #create IC matrix
    x = extrinsics[0]
    y = extrinsics[1]
    z = extrinsics[2]

    I = np.identity(3)
    C = np.array([-x, -y, -z])
    C = C.conj().transpose()
    IC = np.column_stack((I,C))

    #create P matrix
    P = K @ R @ IC
    #normalize for homogenous coordinates
    P = P/P[-1][-1]

    return K, R, IC, P, C


def Angles2R(azimuth, tilt, swing):

    #initialize empty array to create R matrix
    R = np.empty((3,3))

    R[0][0] = -np.cos(azimuth) * np.cos(swing) - np.sin(azimuth) * np.cos(tilt) * np.sin(swing)
    R[0][1] = np.cos(swing) * np.sin(azimuth) - np.sin(swing) * np.cos(tilt) * np.cos(azimuth)
    R[0][2] = -np.sin(swing) * np.sin(tilt)
    R[1][0] = -np.sin(swing) * np.cos(azimuth) + np.cos(swing) * np.cos(tilt) * np.sin(azimuth)
    R[1][1] = np.sin(swing) * np.sin(azimuth) + np.cos(swing) * np.cos(tilt) * np.cos(azimuth)
    R[1][2] = np.cos(swing) * np.sin(tilt)
    R[2][0] = np.sin(tilt) * np.sin(azimuth)
    R[2][1] = np.sin(tilt) * np.cos(azimuth)
    R[2][2] = -np.cos(tilt)

    return R


def distortUV(U, V, intrinsics):


    # assign coefficients
    NU = intrinsics[0]
    NV = intrinsics[1]
    c0U = intrinsics[2]
    c0V = intrinsics[3]
    fx = intrinsics[4]
    fy = intrinsics[5]
    d1 = intrinsics[6]
    d2 = intrinsics[7]
    d3 = intrinsics[8]
    t1 = intrinsics[9]
    t2 = intrinsics[10]

    # calculate distorted coordinates
    # normalize distances
    x = (U - c0U) / fx
    y = (V - c0V) / fy

    # radial distortion
    r2 = x * x + y * y
    fr = 1.0 + d1 * r2 + d2 * (r2 * r2) + d3 * (r2 * (r2 * r2))

    #Tangential distortion
    dx = 2.0 * t1 * x * y + t2 * (r2 + 2.0 * x * x)
    dy = t1 * (r2 + 2.0 * y * y) + 2.0 * t2 * x * y

    #apply correction, answer in chip pixel units
    xd = x * fr + dx
    yd = y * fr + dy
    Ud = xd * fx + c0U
    Vd = yd * fy + c0V

    #determine if points are within image
    #initialize flag that all are acceptable (negative coordinates)
    mask = (Ud < 0) | (Ud > NU) | (Vd < 0) | (Vd > NV)
    Ud[mask] = 0
    Vd[mask] = 0

    #determine if tangential distortion is within range
    #find maximum possible tangential distortion at corners
    Um = np.array((0, 0, NU, NU))
    Vm = np.array((0, NV, NV, 0))

    #normalization
    xm = (Um - c0U) / fx
    ym = (Vm - c0V) / fy
    r2m = xm * xm + ym * ym

    #tangential distortion
    dxm = 2.0 * t1 * xm * ym + t2 * (r2m + 2.0 * xm * xm)
    dym = t1 * (r2m + 2.0 * ym * ym) + 2.0 * t2 * xm * ym

    #find values larger than x and y limits
    flag = np.ones_like(Ud)
    flag[np.where(np.abs(dy) > np.max(np.abs(dym)))] = 0.0
    flag[np.where(np.abs(dx) > np.max(np.abs(dxm)))] = 0.0

    return Ud, Vd, flag


def localTransformExtrinsics(extrinsics, origin:list, coords:str):

    local_origin = np.array((origin[0], origin[1]))
    angle = origin[2]

    if coords == "geo":
        # World to local

        extrinsics[0], extrinsics[1] = localTransformPoints(
            local_origin,
            angle,
            1,
            extrinsics[0],
            extrinsics[1]
        )
        extrinsics[3] = extrinsics[3] + angle

        return extrinsics

    else:
        # local to world
        extrinsics[0], extrinsics[1] = localTransformPoints(
            local_origin,
            angle,
            0,
            extrinsics[0],
            extrinsics[1]
        )
        extrinsics[3] = extrinsics[3] - angle

        return extrinsics


def localTransformPoints(localOrigin, localAngle, flag, Xin, Yin):


#World to Local
    if flag == 1:

        xp = Xin-localOrigin[0]
        yp = Yin-localOrigin[1]

        Xout = xp * np.cos(localAngle) + yp * np.sin(localAngle)
        Yout = yp * np.cos(localAngle) - xp * np.sin(localAngle)

        return Xout, Yout

#Local to World
    if flag == 0:

        Xout = Xin * np.cos(localAngle) - Yin * np.sin(localAngle)
        Yout = Yin * np.cos(localAngle) + Xin * np.sin(localAngle)

        Xout = Xout + localOrigin[0]
        Yout = Yout + localOrigin[1]

        return Xout, Yout


if __name__=='__main__':

    base_folder = "./towerframes/4cams"
    year = '2023'
    camera_list = ['BobA', 'BobB', 'MaryC']
    camera = camera_list

    # check folder paths inside base_folder and save folder name that has substring YYYYMMDD
    collects = os.listdir(base_folder)
    collect_dates_by_year = [collect for collect in collects if year in collect]
    collect_dates = [collect[:8] for collect in collect_dates_by_year]
    collect_dates = [DT.datetime.strptime(date, '%Y%m%d') for date in collect_dates]
    # sort
    collect_dates.sort()

    for single_collect_date in collect_dates:
        inst_dict, grid_dict = jblib.load_inst_and_grid_dict(single_collect_date)

        for i in range(len(camera)):
            _intrinsics, _extrinsics, _rect_year, _rect_month, _rect_day = extract_int_ext(single_collect_date.year,
                                                                                           single_collect_date.month,
                                                                                           single_collect_date.day,
                                                                                           camera[i])

            if i == 0:
                intrinsics = _intrinsics
                extrinsics = _extrinsics
                rect_year = _rect_year
                rect_month = _rect_month
                rect_day = _rect_day
            else:
                intrinsics = np.expand_dims(intrinsics, axis=0)
                extrinsics = np.expand_dims(extrinsics, axis=0)
                _intrinsics = np.expand_dims(_intrinsics, axis=0)
                _extrinsics = np.expand_dims(_extrinsics, axis=0)
                intrinsics = np.append(intrinsics, _intrinsics, axis=1)
                extrinsics = np.append(extrinsics, _extrinsics, axis=1)
                rect_year = np.append(rect_year, _rect_year)
                rect_month = np.append(rect_month, _rect_month)
                rect_day = np.append(rect_day, _rect_day)
                intrinsics = np.squeeze(np.asarray(intrinsics))
                extrinsics = np.squeeze(np.asarray(extrinsics))
        #
        # if len(np.shape(intrinsics)) == 3:
        #     intrinsics = np.squeeze(np.asarray(intrinsics))
        #     extrinsics = np.squeeze(np.asarray(extrinsics))
        #     rect_year = np.squeeze(np.asarray(rect_year)).astype(int)
        #     rect_month = np.squeeze(np.asarray(rect_month)).astype(int)
        #     rect_day = np.squeeze(np.asarray(rect_day)).astype(int)

        xMin, xMax, yMin, yMax, camera_data, rectified_images = rectification_local(intrinsics, extrinsics, rect_year,
                                                                                    rect_month, rect_day, camera)