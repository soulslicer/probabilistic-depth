'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

# data loader for KITTI dataset #
# We will use pykitti module to help to read the image and camera pose #

import pykitti
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import numpy as np
import os
import math
import sys
import glob
import os.path
import scipy.io as sio

import PIL
import torch
import torch.utils.data as data
import torchvision.transforms as tfv_transform
import external.utils_lib.utils_lib as kittiutils
import utils.img_utils as util
import json

import warping.view as View

import torch.nn.functional as F
import torchvision.transforms as transforms
import random
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],\
        'std': [0.229, 0.224, 0.225]}

class ilim_module:
    """Load and parse raw data into a usable format."""

    def __init__(self, base_path, date, drive, **kwargs):
        """Set the path and pre-load calibration data and timestamps."""
        self.dataset = kwargs.get('dataset', 'ilim')
        self.drive = date + '_drive_' + drive + '_' + self.dataset
        self.calib_path = os.path.join(base_path, date)
        self.data_path = os.path.join(base_path, date, self.drive)
        self.leftcam_files = os.listdir(self.data_path + "/left_img")
        self.rightcam_files = os.listdir(self.data_path + "/right_img")
        self.lidar_files = os.listdir(self.data_path + "/lidar")
        self.N = len(self.leftcam_files)
        self.calib = json.load(open(self.data_path + "/calib.json"))

        self.lidar_2_left = np.linalg.inv(np.array(self.calib["left_2_lidar"]))
        self.left_2_right = np.array(self.calib["left_2_right"])
        self.lidar_2_right = np.dot(self.left_2_right, self.lidar_2_left)
        self.left_K = np.array(self.calib["left_P"])[0:3, 0:3]
        self.right_K = np.array(self.calib["right_P"])[0:3, 0:3]

    def __len__(self):
        """Return the number of frames loaded."""
        return self.N

    def get_lidar_files(self):
        return self.lidar_files

    def get_leftcam_files(self):
        return self.leftcam_files

    def get_rightcam_files(self):
        return self.rightcam_files

    def get_lidar_2_leftcam(self):
        return self.lidar_2_left

    def get_lidar_2_rightcam(self):
        return self.lidar_2_right

    def get_leftcam_2_rightcam(self):
        return self.left_2_right

    def get_imu_2_leftcam(self):
        return np.eye(4)

    def get_imu_2_rightcam(self):
        return np.eye(4)

    def get_left_img(self, indx):
        index_str = "%06d" % (indx,)
        return PIL.Image.open(self.data_path + "/left_img/" + index_str + ".png")

    def get_right_img(self, indx):
        index_str = "%06d" % (indx,)
        return PIL.Image.open(self.data_path + "/right_img/" + index_str + ".png")

    def get_lidar(self, indx):
        index_str = "%06d" % (indx,)
        lidarpoints = np.fromfile(self.data_path + "/lidar/" + index_str + ".bin", dtype=np.float32).reshape((-1, 4))
        return lidarpoints

    def get_left_K(self):
        return self.left_K

    def get_right_K(self):
        return self.right_K

    def get_left_size(self):
        return self.get_left_img(0).size

    def get_right_size(self):
        return self.get_right_img(0).size

    def get_pose(self, indx):
        return np.eye(4)

class kitti_module(pykitti.raw):
    """Load and parse raw data into a usable format."""

    def __init__(self, base_path, date, drive, **kwargs):
        super(kitti_module, self).__init__(base_path, date, drive, **kwargs)

    def get_lidar_files(self):
        return self.velo_files

    def get_leftcam_files(self):
        return self.cam2_files

    def get_rightcam_files(self):
        return self.cam3_files

    def get_lidar_2_leftcam(self):
        return self.calib.T_cam2_velo

    def get_lidar_2_rightcam(self):
        return self.calib.T_cam3_velo

    def get_imu_2_leftcam(self):
        return self.calib.T_cam2_imu

    def get_imu_2_rightcam(self):
        return self.calib.T_cam3_imu

    def get_leftcam_2_rightcam(self):
        return np.dot(self.get_imu_2_rightcam(), np.linalg.inv(self.get_imu_2_leftcam()))

    def get_left_img(self, indx):
        return self.get_cam2(indx)

    def get_right_img(self, indx):
        return self.get_cam3(indx)

    def get_lidar(self, indx):
        return self.get_velo(indx)

    def get_left_K(self):
        return self.calib.K_cam2

    def get_right_K(self):
        return self.calib.K_cam3

    def get_left_size(self):
        return self.get_cam2(0).size

    def get_right_size(self):
        return self.get_cam3(0).size

    def get_pose(self, indx):
        return self.oxts[indx].T_w_imu

def normalize_intensity( normalize_paras_):
    '''
    ToTensor(), Normalize()
    '''
    transform_list = [ transforms.ToTensor(),
           transforms.Normalize(**__imagenet_stats)
                       ]
    return transforms.Compose(transform_list)

def to_tensor():
    return transforms.ToTensor()

def get_transform():
    '''
    API to get the transformation.
    return a list of transformations
    '''
    transform_list = normalize_intensity(__imagenet_stats)
    return transform_list

def dMap_to_indxMap(dmap, d_candi):
    '''
    Convert the depth map to the depthIndx map. This is useful if we want to NLL loss to train NN

    Inputs:
    dmap - depth map in 2D. It is a torch.tensor or numpy array
    d_candi - the candidate depths. The output indexMap has the property:
    dmap[i, j] = d_candi[indxMap[i, j]] (not strictly equal, but the depth value in dmap lies
    in the bins defined by d_candi)

    Outputs:
    indxMap - the depth index map. A 2D ndarray
    '''
    assert isinstance(dmap, np.ndarray) or isinstance(dmap, torch.Tensor), \
        'dmap should be a tensor/ndarray'

    if isinstance(dmap, np.ndarray):
        dmap_ = dmap
    else:
        dmap_ = dmap.cpu().numpy()

    indxMap = np.digitize(dmap_, d_candi)

    return indxMap


def _read_libviso_res(filepath, if_filter=False):
    '''
    filepath - file path to the .mat file
    .mat file includes : 'mat_Ts', 'img_paths'
    '''
    mat_info = sio.loadmat(filepath)
    mat_Ts = mat_info['mat_Ts']
    img_paths = mat_info['img_paths']
    img_paths = [str(img_paths[0][i][0]) for i in range(len( img_paths[0]))  ]

    filt_win = 11

    if if_filter:
        import scipy.signal as ssig

        T_traj = mat_Ts[:3, 3, :]
        Tx_traj, Ty_traj, Tz_traj = T_traj[0, :], T_traj[1, :], T_traj[2, :]
        b,a = ssig.butter(4, 1/filt_win, 'low')
        Tx_traj_filt = ssig.filtfilt(b,a, Tx_traj, )
        Ty_traj_filt = ssig.filtfilt(b,a, Ty_traj, )
        Tz_traj_filt = ssig.filtfilt(b,a, Tz_traj, )

        mat_Ts[0, 3, :] = Tx_traj_filt
        mat_Ts[1, 3, :] = Ty_traj_filt
        mat_Ts[2, 3, :] = Tz_traj_filt

    return mat_Ts, img_paths

def _read_split_file( filepath):
    '''
    Read data split txt file provided by KITTI dataset authors
    '''
    with open(filepath) as f:
        trajs = f.readlines()
    trajs = [ x.strip() for x in trajs ]

    return trajs

def _read_IntM_from_pdata( p_data,  out_size = None,   mode = "left",   crop_amt = [1., 1.]):
    '''
    Get the intrinsic camera info from pdata
    raw_img_size - [width, height]
    '''

    IntM = np.zeros((4,4))

    if mode == "left":
        raw_img_size = p_data.get_left_size()
        IntM = p_data.get_left_K().copy()
    elif mode == "right":
        raw_img_size = p_data.get_right_size()
        IntM = p_data.get_right_K().copy()

    width = int( raw_img_size[0] )
    height = int( raw_img_size[1])

    # HACK BECAUSE OF CROPPING
    IntM[0, 0] *= crop_amt[0]
    width /= crop_amt[0]
    IntM[1, 1] *= crop_amt[1]
    height /= crop_amt[1]

    focal_length = np.mean([IntM[0,0], IntM[1,1]])
    h_fov = math.degrees(math.atan(IntM[0, 2] / IntM[0, 0]) * 2)
    v_fov = math.degrees(math.atan(IntM[1, 2] / IntM[1, 1]) * 2)

    if p_data.mode == "kitti":
        if out_size is not None: # the depth map is re-scaled #
            camera_intrinsics = np.zeros((3,4))
            pixel_width, pixel_height = out_size[0], out_size[1]
            camera_intrinsics[2,2] = 1.
            camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(h_fov/2.0))
            camera_intrinsics[0,2] = pixel_width/2.0
            camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(v_fov/2.0))
            camera_intrinsics[1,2] = pixel_height/2.0

            IntM = camera_intrinsics
            focal_length = pixel_width / width * focal_length
            width, height = pixel_width, pixel_height
    elif p_data.mode == "ilim":
        if out_size is not None:
            pixel_width, pixel_height = out_size[0], out_size[1]
            xscale = float(pixel_width)/float(width)
            yscale = float(pixel_height)/float(height)
            IntM[0,:] *= xscale
            IntM[1,:] *= yscale
            focal_length = xscale * focal_length
            width, height = pixel_width, pixel_height

    # In scanenet dataset, the depth is perperdicular z, not ray distance #
    pixel_to_ray_array = View.normalised_pixel_to_ray_array(\
            width= int(width), height= int(height), hfov = h_fov, vfov = v_fov,
            normalize_z = True)

    pixel_to_ray_array_2dM = np.reshape(np.transpose( pixel_to_ray_array, axes= [2,0,1] ), [3, -1])
    pixel_to_ray_array_2dM = torch.from_numpy(pixel_to_ray_array_2dM.astype(np.float32))
    cam_intrinsic = {\
            'hfov': h_fov, 'vfov': v_fov,
            'unit_ray_array': pixel_to_ray_array,
            'unit_ray_array_2D': pixel_to_ray_array_2dM,
            'intrinsic_M_cuda': torch.from_numpy(IntM[:3,:3].astype(np.float32)),
            'focal_length': focal_length,
            'intrinsic_M': IntM}
    return cam_intrinsic

def get_paths(traj_indx, database_path_base = '/datasets/kitti', scene_names = None, mode = 'train', t_win = 2):
    '''
    Return the training info for one trajectory
    Assuming:

    (1) the kitti data set is organized as
    /path_to_kitti/rawdata
    /path_to_kitti/train
    /path_to_kitti/val
    where rawdata folder contains the raw kitti data,
    train and val folders contains the GT depth

    (2) the depth frames for one traj. is always nimg - 10 (ignoring the first and last 5 frames)

    (3) we will use image 02 data (the left camera ?)

    Inputs:
    - traj_indx : the index (in the globbed array) of the trajectory
    - databse_path_base : the path to the database
    - split_txt : the split txt file path, including the name of trajectories for trianing/testing/validation
    - mode: 'train' or 'val' or 'test'

    Outputs:
    - n_traj : the # of trajectories in the set defined in split_txt
    - pykitti_dataset : the dataset object defined in pykitti. It is used to get the input images and camera poses
    - dmap_paths: the list of paths for the GT dmaps, same length as dataset
    - poses: list of camera poses, corresponding to dataset and dmap_paths

    '''

    n_traj = len( scene_names )

    assert traj_indx < n_traj, 'traj_indx should smaller than the scene length'

    basedir = database_path_base
    sceneName = scene_names[traj_indx]
    name_contents = sceneName.split('_')
    date = name_contents[0] + '_' + name_contents[1] + '_' + name_contents[2]
    drive = name_contents[4]
    dataset = name_contents[5]
    #print("Loaded Scene: " + str(sceneName))

    # Load Type (Need a better check for this)
    if dataset == "sync":
        p_data_full = kitti_module(basedir, date, drive)
        mode = "kitti"
    elif dataset == "ilim":
        p_data_full = ilim_module(basedir, date, drive)
        mode = "ilim"

    # Kitti
    if mode == "kitti":

        # assume: the depth frames for one traj. is always nimg - 10 (ignoring the first and last 5 frames)
        nimg = len(p_data_full)
        fsize = t_win*2 + 1
        p_data = kitti_module(basedir, date, drive, frames= range(0, nimg-0))

        nimg = len(p_data)
        dmap_paths = [[],[]]

        poses = []
        for i_img in range(nimg):
            poses.append(p_data.get_pose(i_img))

        intrin_path = ''
        setattr(p_data, 'mode', 'kitti')
        setattr(p_data, 'scene_name', sceneName)
        return n_traj, p_data, dmap_paths, poses, intrin_path

    # ILIM
    elif mode == "ilim":
        p_data = p_data_full
        setattr(p_data, 'mode', 'ilim')
        setattr(p_data, 'scene_name', sceneName)
        return n_traj, p_data, [], [], ''
        
def _read_left_img(p_data, indx, img_size = None, no_process= False, only_resize = False):
    '''
    Read image and process
    '''
    proc_img = get_transform()
    if no_process:
        img = p_data.get_left_img(indx)
        width, height = img.size
    else:
        if img_size is not None:
            img = p_data.get_left_img(indx)
            img = img.resize( img_size, PIL.Image.BICUBIC )
        else:
            img = p_data.get_left_img(indx)

        width, height = img.size
        if not only_resize:
            img = proc_img(img)

    raw_sz = (width, height)
    return img,  raw_sz

def _read_right_img(p_data, indx, img_size = None, no_process= False, only_resize = False):
    '''
    Read image and process
    '''
    proc_img = get_transform()
    if no_process:
        img = p_data.get_right_img(indx)
        width, height = img.size
    else:
        if img_size is not None:
            img = p_data.get_right_img(indx)
            img = img.resize( img_size, PIL.Image.BICUBIC )
        else:
            img = p_data.get_right_img(indx)

        width, height = img.size
        if not only_resize:
            img = proc_img(img)

    raw_sz = (width, height)
    return img,  raw_sz

def _read_dimg(path, img_size = None, no_process= False, only_resize = False):
    '''
    Read image and process
    '''

    if os.path.exists(path):
        proc_img = get_transform()
        if no_process:
            img = PIL.Image.open(path)
            width, height = img.size
        else:
            if img_size is not None:
                img = PIL.Image.open(path).convert('RGB')
                img = img.resize( img_size, PIL.Image.BICUBIC )
            else:
                img = PIL.Image.open(path).convert('RGB')
            width, height = img.size
            if not only_resize:
                img = proc_img(img)

        raw_sz = (width, height)
        return img,  raw_sz
    else:
        return -1, -1

class KITTI_dataset(data.Dataset):

    def __init__(self, training, p_data, dmap_seq_paths, poses, intrin_path,
                 img_size = [1248,380], digitize = False, d_candi = None, d_candi_up = None, resize_dmap = None, if_process = True,
                 crop_w = 384, velodyne_depth = True, pytorch_scaling = False, cfg = None):

        '''
        inputs:

        traning - if for training or not

        p_data - a pykitti.raw object, returned by get_paths()

        dmap_seq_paths - list of dmaps, returned by get_paths()

        poses - list of posesc

        d_candi - the candidate depths
        digitize (Optional; False) - if digitize the depth map using d_candi

        resize_dmap - the scale for re-scaling the dmap the GT dmap size would be self.img_size * resize_dmap

        if_process - if do post process

        crop_w - the cropped image width. We will crop the central region with wdith crop_w

        '''

        if crop_w is not None:
            assert (img_size[0] - crop_w )%2 ==0 and crop_w%4 ==0
            assert resize_dmap is not None

        self.pytorch_scaling = pytorch_scaling
        self.velodyne_depth = velodyne_depth
        self.p_data = p_data
        self.dmap_seq_paths = dmap_seq_paths
        self.poses = poses
        self.training = training
        self.intrin_path = intrin_path # reserved
        self.cfg = cfg
        self.counter = 0

        self.to_gray = tfv_transform.Compose( [tfv_transform.Grayscale(), tfv_transform.ToTensor()])
        self.d_candi = d_candi
        self.digitize = digitize

        self.crop_w = crop_w

        if digitize:
            self.label_min = 0
            self.label_max = len(d_candi) - 1

        self.dup4_candi = d_candi_up
        self.dup4_label_min = 0
        self.dup4_label_max = len(self.dup4_candi) -1

        self.resize_dmap = resize_dmap

        self.img_size = img_size # the raw input image size (used for resizing the input images)

        self.if_preprocess = if_process

        if crop_w is not None:
            self.crop_amt = [self.img_size[0]/self.crop_w, 1.]
            if self.crop_amt[0] != 2: raise Exception('Check this')
        else:
            self.crop_amt = [1., 1.]

        self.setup_instrinsics()

    def setup_instrinsics(self):
        if self.crop_w is None:
            left_cam_intrinsics = self.get_cam_intrinsics(None, "left")
            right_cam_intrinsics = self.get_cam_intrinsics(None, "right")
        else:
            width_ = int(self.crop_w * self.resize_dmap)
            height_ = int(float( self.img_size[1] * self.resize_dmap))
            img_size_ = np.array([width_, height_], dtype=int)
            left_cam_intrinsics = self.get_cam_intrinsics(img_size = img_size_ , mode = "left", crop_amt = self.crop_amt)
            right_cam_intrinsics = self.get_cam_intrinsics(img_size=img_size_, mode = "right", crop_amt = self.crop_amt)

        self.left_cam_intrinsics = left_cam_intrinsics
        self.right_cam_intrinsics = right_cam_intrinsics

    def get_cam_intrinsics(self,  img_size = None,  mode = "left",  crop_amt = [1., 1.]):
        '''
        Get the camera intrinsics
        '''
        if img_size is not None:
            width = img_size[0]
            height = img_size[1]
        else:
            if self.resize_dmap is None:
                width = self.img_size[0]
                height = self.img_size[1]
            else:
                width = int(float( self.img_size[0] * self.resize_dmap))
                height = int(float( self.img_size[1] * self.resize_dmap))

        cam_intrinsics = _read_IntM_from_pdata( self.p_data, out_size = [width, height], mode = mode, crop_amt = crop_amt)

        return cam_intrinsics

    def generate_item(self, indx, mode="left"):
        self.counter += 1
        pass

        if not self.velodyne_depth:
            if mode == "left":
                dmap_path = self.dmap_seq_paths[0][indx]
            elif mode == "right":
                dmap_path = self.dmap_seq_paths[1][indx]

        proc_normalize = get_transform()
        proc_totensor = to_tensor()

        # IMU to camera #
        intr_raw = None
        raw_img_size = None #[1226, 370]
        if mode == "left":
            M_imu2cam = self.p_data.get_imu_2_leftcam()
            M_velo2cam = self.p_data.get_lidar_2_leftcam()
            intr_raw = self.p_data.get_left_K()
            raw_img_size = self.p_data.get_left_size()
        elif mode == "right":
            M_imu2cam = self.p_data.get_imu_2_rightcam()
            M_velo2cam = self.p_data.get_lidar_2_rightcam()
            intr_raw = self.p_data.get_right_K()
            raw_img_size = self.p_data.get_right_size()

        # Velodyne or Depth Load
        if self.velodyne_depth:
            if indx >= len(self.p_data.get_lidar_files()):
                string = "Index accessed wrongly: " + str(indx) + " " + str(len(self.p_data.get_lidar_files()))
                raise Exception(string)
            velodata = self.p_data.get_lidar(indx) # [N x 4] [We could clean up the low intensity ones here!]
            intr_raw_append = np.append(intr_raw, np.array([[0, 0, 0]]).T, axis=1)
            velodata[:,3] = 1.
            dmap_raw = None
        else:
            dmap_raw = _read_dimg(dmap_path, no_process=True)[0] #[1226, 370]
            scale_factor = 256.

        # Read RGB
        if mode == "left":
            img = _read_left_img(self.p_data, indx, no_process=True)[0]
        elif mode == "right":
            img = _read_right_img(self.p_data, indx, no_process=True)[0]
        img = img.resize(self.img_size, PIL.Image.BILINEAR)
        if self.resize_dmap is not None:
            if not self.pytorch_scaling:
                img_dw = img.resize(
                    [int(self.img_size[0] * self.resize_dmap), int(self.img_size[1] * self.resize_dmap)],
                    PIL.Image.BILINEAR)
            else:
                img_pytorch = proc_totensor(img)
                sfactor = self.resize_dmap
                img_pytorch_dw_bl = F.interpolate(img_pytorch.unsqueeze(0), scale_factor=sfactor, mode='bilinear').squeeze(0)
                temp = (img_pytorch_dw_bl.numpy().transpose(1,2,0)*255).astype(np.uint8)
                img_dw = PIL.Image.fromarray(temp)
        else:
            img_dw = None
        img_gray = self.to_gray(img)

        # read GT depth map (if available) #
        if dmap_raw is not -1:

            # Loading Depth
            if not self.velodyne_depth:

                if self.resize_dmap is not None:

                    # Convert Image to [256, 768]
                    dmap_imgsize = dmap_raw.resize([self.img_size[0], self.img_size[1]], PIL.Image.NEAREST)

                    if not self.pytorch_scaling:

                        # Resize Downsample (Added)
                        dmap_size = [int(float(dmap_imgsize.width) * self.resize_dmap),
                                     int(float(dmap_imgsize.height) * self.resize_dmap)]
                        dmap_raw_bilinear_dw = dmap_imgsize.resize(dmap_size, PIL.Image.BILINEAR)
                        dmap_raw = dmap_imgsize.resize(dmap_size, PIL.Image.NEAREST)
                        #dmap_mask = dmap_mask_imgsize.resize(dmap_size, PIL.Image.NEAREST)

                        # Convert to Tensor
                        dmap_imgsize = proc_totensor(dmap_imgsize)[0, :, :].float() / scale_factor
                        dmap_rawsize = proc_totensor(dmap_raw)[0, :, :].float() / scale_factor
                    else:

                        # Convert to Tensor
                        dmap_imgsize = proc_totensor(dmap_imgsize)[0, :, :].float() / scale_factor

                        # Resize using Pytorch
                        dmap_size = [int(float(dmap_imgsize.shape[0]) * self.resize_dmap),
                                     int(float(dmap_imgsize.shape[1]) * self.resize_dmap)]
                        dmap_raw_bilinear_dw = F.interpolate(dmap_imgsize.unsqueeze(0).unsqueeze(0), size=dmap_size, mode='bilinear').squeeze(0).squeeze(0)
                        dmap_raw = F.interpolate(dmap_imgsize.unsqueeze(0).unsqueeze(0), size=dmap_size, mode='nearest').squeeze(0).squeeze(0)
                        dmap_rawsize = dmap_raw

                # Build Mask (Added)
                dmap_mask_imgsize = np.array(dmap_imgsize, dtype=int).astype(np.float32) < 0.01
                dmap_mask = np.array(dmap_rawsize, dtype=int).astype(np.float32) < 0.01

                # Convert to Tensor
                if not self.pytorch_scaling:
                    dmap_raw = proc_totensor(dmap_raw)[0, :, :]  # single-channel for depth map
                    dmap_raw = dmap_raw.float() / scale_factor  # scale to meter
                    dmap_raw_bilinear_dw = proc_totensor(dmap_raw_bilinear_dw)[0, :, :]
                    dmap_raw_bilinear_dw = dmap_raw_bilinear_dw.float() / scale_factor
                else:
                    pass

            # Velodyne Depth
            else:

                # Generate Sizes
                small_size = [int(float(self.img_size[0]) * self.resize_dmap),
                             int(float(self.img_size[1]) * self.resize_dmap)]
                large_size = self.img_size

                # Generate Intr
                large_intr = util.intr_scale(intr_raw_append, raw_img_size, large_size)
                small_intr = util.intr_scale(intr_raw_append, raw_img_size, small_size)

                # Generate Depth maps
                if not self.cfg.lidar.enabled:
                    large_params = {"filtering": 2, "upsample": 0}
                else:
                    large_params = self.cfg.lidar
                dmap_large = kittiutils.generate_depth(velodata, large_intr, M_velo2cam, large_size[0], large_size[1], large_params)
                #small_params = {"filtering": 0, "upsample": 0}
                #dmap_small = kittiutils.generate_depth(velodata, small_intr, M_velo2cam, small_size[0], small_size[1], small_params)

                # Hack
                #print("Hacked Small DMap. And Hacked the RGB Scaling Mode")
                #dmap_small = F.interpolate(torch.Tensor(dmap_large).unsqueeze(0).unsqueeze(0), size=[small_size[1], small_size[0]], mode='nearest').squeeze(0).squeeze(0).numpy()
                #dmap_small = F.max_pool2d(torch.Tensor(dmap_large).unsqueeze(0).unsqueeze(0), 4).squeeze(0).squeeze(0).numpy()
                #dmap_small = F.adaptive_max_pool2d(torch.Tensor(dmap_large).unsqueeze(0).unsqueeze(0), [small_size[1], small_size[0]]).squeeze(0).squeeze(0).numpy()
                dmap_small = util.minpool(torch.Tensor(dmap_large).unsqueeze(0).unsqueeze(0), 4, 1000).squeeze(0).squeeze(0).numpy()

                # Change names
                dmap_imgsize = dmap_large
                dmap_raw = dmap_small

                # Build Mask (Added)
                dmap_mask_imgsize = dmap_imgsize < 0.01
                dmap_mask = dmap_raw < 0.01

                # Convert to Torch
                dmap_imgsize = torch.Tensor(dmap_imgsize)
                dmap_raw = torch.Tensor(dmap_raw)
                dmap_raw_bilinear_dw = dmap_raw.clone()
                dmap_rawsize = dmap_raw.clone()

            # Apply Mask
            if self.resize_dmap is None:
                dmap_rawsize = dmap_raw
            dmap_mask = ~(proc_totensor(dmap_mask) > 0)
            dmap_mask_imgsize = ~(proc_totensor(dmap_mask_imgsize) > 0)
            dmap_raw = dmap_raw * dmap_mask.squeeze().type_as(dmap_raw)
            dmap_raw_bilinear_dw = dmap_raw_bilinear_dw * dmap_mask.squeeze().type_as(dmap_raw)
            dmap_imgsize = dmap_imgsize * dmap_mask_imgsize.squeeze().type_as(dmap_imgsize)

            # Digitize
            if self.digitize:
                # digitize the depth map #
                dmap = dMap_to_indxMap(dmap_raw, self.d_candi)
                dmap[dmap >= self.label_max] = self.label_max # We could set this to min too so it get ignored in clamp
                dmap[dmap <= self.label_min] = self.label_min
                dmap = torch.from_numpy(dmap)
                dmap_imgsize_digit = dMap_to_indxMap(dmap_imgsize, self.d_candi)
                dmap_imgsize_digit[dmap_imgsize_digit >= self.label_max] = self.label_max
                dmap_imgsize_digit[dmap_imgsize_digit <= self.label_min] = self.label_min
                dmap_imgsize_digit = torch.from_numpy(dmap_imgsize_digit)
                dmap_up4_imgsize_digit = dMap_to_indxMap(dmap_imgsize, self.dup4_candi)
                dmap_up4_imgsize_digit[dmap_up4_imgsize_digit >= self.dup4_label_max] = self.dup4_label_max
                dmap_up4_imgsize_digit[dmap_up4_imgsize_digit <= self.dup4_label_min] = self.dup4_label_min
                dmap_up4_imgsize_digit = torch.from_numpy(dmap_up4_imgsize_digit)
            else:
                dmap = dmap_raw
                dmap_imgsize_digit = dmap_imgsize
                dmap_up4_imgsize_digit = dmap_imgsize

        # RGB to Pytorch
        if self.if_preprocess:
            img = proc_normalize(img)
            if self.resize_dmap is not None:
                img_dw = proc_normalize(img_dw)
        else:
            proc_totensor = tfv_transform.ToTensor()
            img = proc_totensor(img)
            if self.resize_dmap is not None:
                img_dw = proc_totensor(img_dw)

        # Cropping
        if self.crop_w is not None:
            side_crop = int((self.img_size[0] - self.crop_w) / 2)
            side_crop_dw = int(side_crop * self.resize_dmap)

            img_size = self.img_size
            img = img[:, :, side_crop: img_size[0] - side_crop]
            img_dw = img_dw[:, :, side_crop_dw: img_dw.shape[-1] - side_crop_dw]

            img_gray = img_gray[:, :, side_crop: img_size[0] - side_crop]

            if dmap_raw is not -1:
                dmap = dmap[:, side_crop_dw: (dmap.shape[1] - side_crop_dw)]
                dmap_raw = dmap_raw[:, side_crop_dw: (dmap_raw.shape[1] - side_crop_dw)]
                dmap_raw_bilinear_dw = dmap_raw_bilinear_dw[
                                       :, side_crop_dw: (dmap_raw_bilinear_dw.shape[1] - side_crop_dw)]
                dmap_rawsize = dmap_rawsize[:, side_crop: (dmap_rawsize.shape[1] - side_crop)]
                dmap_imgsize = dmap_imgsize[:, side_crop: (dmap_imgsize.shape[1] - side_crop)]

                dmap_imgsize_digit = dmap_imgsize_digit[
                                     :, side_crop: (dmap_imgsize_digit.shape[1] - side_crop)]

                dmap_up4_imgsize_digit = dmap_up4_imgsize_digit[:,
                                         side_crop: (dmap_up4_imgsize_digit.shape[1] - side_crop)]

                dmap_mask = dmap_mask[:, :, side_crop_dw: (dmap_mask.shape[-1] - side_crop_dw)]
                dmap_mask_imgsize = dmap_mask_imgsize[
                                    :, :, side_crop: (dmap_mask_imgsize.shape[-1] - side_crop)]

        # Read Params
        if len(self.poses):
            extM = np.matmul(M_imu2cam, np.linalg.inv(self.poses[indx]))
        else:
            extM = np.eye(4)
        # image path #
        scene_path = self.p_data.calib_path
        if mode == "left":
            img_path = self.p_data.get_leftcam_files()[indx]
        elif mode == "right":
            img_path = self.p_data.get_rightcam_files()[indx]

        return {'img': img.unsqueeze_(0),
                'img_dw': img_dw.unsqueeze_(0),
                'dmap': dmap.unsqueeze_(0) if dmap_raw is not -1 else -1,
                'dmap_raw': dmap_raw.unsqueeze_(0) if dmap_raw is not -1 else -1,
                 #'dmap_raw_bilinear_dw': dmap_raw_bilinear_dw.unsqueeze_(0) if dmap_raw is not -1 else -1,
                 #'dmap_rawsize': dmap_rawsize.unsqueeze_(0) if dmap_raw is not -1 else -1,
                'dmap_imgsize': dmap_imgsize.unsqueeze_(0) if dmap_raw is not -1 else -1,
                'dmap_imgsize_digit': dmap_imgsize_digit.unsqueeze_(0) if dmap_raw is not -1 else -1,
                'dmap_up4_imgsize_digit': dmap_up4_imgsize_digit.unsqueeze_(0) if dmap_raw is not -1 else -1,
                'dmap_mask': dmap_mask.unsqueeze_(0).type_as(dmap) if dmap_raw is not -1 else -1,
                'dmap_mask_imgsize': dmap_mask_imgsize.unsqueeze_(0).type_as(dmap) if dmap_raw is not -1 else -1,
                'img_gray': img_gray.unsqueeze_(0),
                'extM': extM,
                'scene_path': scene_path,
                'img_path': img_path, }

    def __getitem__(self, indx):
        '''
        outputs:
        img, dmap, extM, scene_path , as entries in a dic.
        '''

        try:
            left_item = self.generate_item(indx, "left")
            right_item = self.generate_item(indx, "right")
            T_left2right = self.p_data.get_leftcam_2_rightcam()

            return {"left_camera": left_item, "right_camera": right_item, "T_left2right": T_left2right, "success": True}
        except:
            import traceback
            traceback.print_exc()
            return {"success": False}

    def __len__(self):
        return len(self.p_data)

    def set_paths(self, p_data, dmap_seq_paths, poses):
        '''
        set the p_data, poses and dmaps paths
        p_data - a pykitti.raw object, returned by get_paths()
        dmap_seq_paths - list of dmaps, returned by get_paths()
        poses - list of posesc
        '''
        print("Loaded: " + str(p_data.scene_name))
        self.p_data = p_data
        self.dmap_seq_paths = dmap_seq_paths
        self.poses = poses

        # # Hardcode test (If doing both at once?)
        # if p_data.mode == "kitti":
        #     self.img_size = [768, 256]
        #     self.crop_w = 384
        # elif p_data.mode == "ilim":
        #     self.img_size = [384, 256]
        #     self.crop_w = None

        # if self.crop_w is not None:
        #     self.crop_amt = [self.img_size[0]/self.crop_w, 1.]
        #     if self.crop_amt[0] != 2: raise Exception('Check this')
        # else:
        #     self.crop_amt = [1., 1.]

        # self.setup_instrinsics()