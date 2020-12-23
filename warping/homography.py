'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''


'''
Homography warping module: 
'''
import numpy as np
from warping import view as View
import torch.nn.functional as F
import torch

import time
import math

def get_rel_extM_list(abs_extMs, rel_extMs, t_win_r = 2):
    '''
    Get the list of relative camera pose
    input: 

    abs_extMs - the list of absolute camera pose
    if abs_extMs[idx]==-1, then this idx time stamp has invalid pose
    rel_extMs - list of list rel camera poses, 
    rel_extMs[idx] = [zeors] by default, if invalid pose, then =zeros

    '''
    n_abs_extMs = len(abs_extM)
    for idx_extM in range(n_abs_extMs):
        if idx_extM + 1 - t_win_r < 0:
            continue
        elif abs_extMs[idx_extM] ==-1:
            continue
        else:
            pass

def points3D_to_opticalFlow(Points3D, view_ref, view_src):
    '''
    Get the optical flow from ref view to src view
    This function is used for validation for the warping function : back_warp()
    Inputs: 
        Points3D - points in 3D in the world coordiate 
        view_ref , view_src : the camera views including the camera position and lookat
    Outputs: 
        optical_flow - the optical flow from ref view to src view
    '''
    import mdataloader.sceneNet_calculate_optical_flow as sceneNet_opticalFlow
    return sceneNet_opticalFlow.optical_flow( Points3D, view_ref, view_src)

def img_dis_L2(img0, img1):
    assert isinstance(img0, torch.Tensor) and isinstance(img1, torch.Tensor), 'inputs should be Torch tensors'
    diff_img = torch.sum(((img0 - img1)**2).squeeze(), 0)
    return diff_img 

def img_dis_L2_diffmask(img0, img1):
    '''
    also return the distance in mask, to deal with image boundaries
    NOTE: Assuming the mask image is the first feature channel !
    '''
    assert isinstance(img0, torch.Tensor) and isinstance(img1, torch.Tensor), 'inputs should be Torch tensors'
    feat_diff_img = torch.sum(((img0[0, 1::, ... ] - img1[0, 1::, ... ])**2).squeeze(), 0)
    mask_diff_img = (img0[0, 0, ...] - img1[0, 0, ...])**2 

    return feat_diff_img, mask_diff_img

def img_dis_L2_mask(img0, img1):
    '''
    also return the warped mask, viewed from img1
    NOTE: Assuming the mask image is the first feature channel !
    '''
    assert isinstance(img0, torch.Tensor) and isinstance(img1, torch.Tensor), 'inputs should be Torch tensors'
    feat_diff_img = torch.sum(((img0[0, 1::, ... ] - img1[0, 1::, ... ])**2).squeeze(), 0)
    mask_diff_img = (img0[0, 0, ...] - img1[0, 0, ...])**2 

    return feat_diff_img, mask_diff_img, img1[0, 0, ...]

def img_dis_L2_pard(img0, img1):
    diff_img = torch.sum( (img0 - img1)**2, 1)
    return diff_img

def img_dis_L1_pard(img0, img1):
    diff_img = torch.sum( torch.abs(img0 - img1), 1)
    return diff_img

def debug_writeVolume(vol, vmin, vmax):
    '''
    vol : D x H x W
    '''
    import matplotlib.pyplot as plt
    for idx in range(vol.shape[0]):
        slice_img = vol[idx,:, :]
        plt.imsave('vol_%03d.png'%(idx), 
                   slice_img, vmin=vmin, vmax=vmax)

def est_swp_volume_v4(feat_img_ref, feat_img_src, 
                      d_candi, R,t, cam_intrinsic,
                      costV_sigma,  
                      feat_dist = 'L2',
                      debug_ipdb = False):
    r'''
    feat_img_ref - NCHW tensor
    feat_img_src - NVCHW tensor.  V is for different views
    R, t - R[idx_view, :, :] - 3x3 rotation matrix
           t[idx_view, :] - 3x1 transition vector
    '''
    device = feat_img_ref.device
    H, W, D = feat_img_ref.shape[2], feat_img_ref.shape[3], len(d_candi)
    costV = torch.zeros(1, D, H, W).to(device)

    IntM_tensor = cam_intrinsic['intrinsic_M_cuda'].to(device) # intrinsic matrix 3x3 on GPU
    P_ref_cuda = cam_intrinsic['unit_ray_array_2D'].to(device) # unit ray array in matrix form on GPU
    d_candi_cuda = torch.from_numpy(d_candi.astype(np.float32)).to(device)

    for idx_view in range(feat_img_src.shape[1]): # Iterate each image
        # Get term1 #
        term1 = IntM_tensor.matmul(t[idx_view, :]).reshape(3,1)
        # Get term2 #
        term2 = IntM_tensor.matmul(R[idx_view, :, :]).matmul(P_ref_cuda)
        feat_img_src_view = feat_img_src[:, idx_view, :, :, :]
        feat_img_src_view_repeat = feat_img_src_view.repeat(len(d_candi), 1, 1, 1)

        feat_img_src_view_warp_par_d = \
        _back_warp_homo_parallel(feat_img_src_view_repeat, d_candi_cuda, term1, term2, cam_intrinsic, H, W)

        if feat_dist == 'L2':
            costV[0, :, :, :] = costV[0,:,:,:] + img_dis_L2_pard(feat_img_src_view_warp_par_d, feat_img_ref) / costV_sigma
        elif feat_dist == 'L1':
            costV[0, :, :, :] = costV[0,:,:,:] + img_dis_L1_pard(feat_img_src_view_warp_par_d, feat_img_ref) / costV_sigma
        else:
            raise Exception('undefined metric for feature distance ...')

    return costV

def warp_feature(feat_img_src,
                 d_candi, R,t, cam_intrinsic):
    r'''
    feat_img_src - NVCHW tensor.  V is for different views
    R, t - R[idx_view, :, :] - 3x3 rotation matrix
           t[idx_view, :] - 3x1 transition vector
    '''
    if feat_img_src.shape[0] != 1:
        raise Exception("Warped Accum Error")
    device = feat_img_src.device
    H, W, D = feat_img_src.shape[3], feat_img_src.shape[4], len(d_candi)

    IntM_tensor = cam_intrinsic['intrinsic_M_cuda'].to(device) # intrinsic matrix 3x3 on GPU
    P_ref_cuda = cam_intrinsic['unit_ray_array_2D'].to(device) # unit ray array in matrix form on GPU
    d_candi_cuda = torch.from_numpy(d_candi.astype(np.float32)).to(device)

    warped_accum = torch.zeros(feat_img_src.shape).to(feat_img_src.device)
    for idx_view in range(feat_img_src.shape[1]): # Iterate each image
        # Get term1 #
        term1 = IntM_tensor.matmul(t[idx_view, :]).reshape(3,1)
        # Get term2 #
        term2 = IntM_tensor.matmul(R[idx_view, :, :]).matmul(P_ref_cuda)
        feat_img_src_view = feat_img_src[:, idx_view, :, :, :]
        feat_img_src_view_repeat = feat_img_src_view.repeat(len(d_candi), 1, 1, 1)

        feat_img_src_view_warp_par_d = \
        _back_warp_homo_parallel(feat_img_src_view_repeat, d_candi_cuda, term1, term2, cam_intrinsic, H, W)

        for i in range(0, len(d_candi)):
            warped_accum[0, idx_view, i, :, :] = feat_img_src_view_warp_par_d[i,i,:,:]

    return warped_accum

def _back_warp_homo_parallel(img_src, D, term1, term2, cam_intrinsics, H, W, debug_inputs = None ):
    r'''
    Do the warpping for the src. view analytically using homography, given the
    depth d for the reference view: 
    p_src ~ term1  + term2 * d

    inputs:
    term1, term2 - 3 x n_pix matrix 
    P_ref - The 2D matrix form for the unit_array for the camera
    D - candidate depths. A tensor array on GPU

    img_src_warpped - warpped src. image 
    '''
    n_d = len(D)
    device = img_src.device
    term2_cp = term2.repeat(n_d, 1, 1)

    P_src = term1.unsqueeze(0) + term2_cp * D.reshape(n_d,1,1)
    P_src = P_src / (P_src[:, 2, :].unsqueeze(1)  + 1e-10 )

    src_coords = torch.FloatTensor(n_d, H, W, 2).to(device)

    src_coords[:,:,:,0] = P_src[:, 0, :].reshape(n_d, H, W)
    src_coords[:,:,:,1] = P_src[:, 1, :].reshape(n_d, H, W)
    u_center, v_center = cam_intrinsics['intrinsic_M'][0,2], cam_intrinsics['intrinsic_M'][1,2]
    src_coords[:,:,:,0] = (src_coords[:,:,:,0] - u_center) / u_center
    src_coords[:,:,:,1] = (src_coords[:,:,:,1] - v_center) / v_center
    img_src_warpped = F.grid_sample(img_src, src_coords,mode='bilinear', padding_mode='zeros') 
    return img_src_warpped

def _back_warp_homo(img_src, d, term1, term2, cam_intrinsics, H, W,
         debug_inputs = None ):
    r'''
    Do the warpping for the src. view analytically using homography, given the
    depth d for the reference view: 
    p_src ~ term1  + term2 * d

    inputs:
    term1, term2 - 3 x n_pix matrix 
    P_ref - The 2D matrix form for the unit_array for the camera
    d - candidate depth

    img_src_warpped - warpped src. image 
    '''
    P_src = term1 + term2 * d
    P_src = P_src / P_src[2,:]
    u_center, v_center = cam_intrinsics['intrinsic_M'][0,2], cam_intrinsics['intrinsic_M'][1,2]
    u_coords, v_coords = P_src[0,:], P_src[1, :]

    u_coords = (u_coords - u_center) / u_center # to range [-1, 1]
    v_coords = (v_coords - v_center) / v_center

    u_coords = torch.reshape(u_coords, [H, W]) 
    v_coords = torch.reshape(v_coords, [H, W])
    src_coords = torch.stack((u_coords, v_coords), dim=2).unsqueeze(0)
    img_src_warpped = F.grid_sample( img_src, src_coords)

    return img_src_warpped

def _set_vol_border( vol, border_val ):
    '''
    inputs:
    vol - a torch tensor in 3D: N x C x D x H x W
    border_val - a float, the border value
    '''
    vol_ = vol + 0.
    vol_[:, :, 0, :, :] = border_val
    vol_[:, :, :, 0, :] = border_val
    vol_[:, :, :, :, 0] = border_val
    vol_[:, :, -1, :, :] = border_val
    vol_[:, :, :, -1, :] = border_val
    vol_[:, :, :, :, -1] = border_val

    return vol_

def _set_vol_border_v0( vol, border_val ):
    '''
    inputs:
    vol - a torch tensor in 3D: N x C x D x H x W
    border_val - a float, the border value
    '''
    vol_ = vol 
    vol_[:, :, 0, :, :] = border_val
    vol_[:, :, :, 0, :] = border_val
    vol_[:, :, :, :, 0] = border_val
    vol_[:, :, -1, :, :] = border_val
    vol_[:, :, :, -1, :] = border_val
    vol_[:, :, :, :, -1] = border_val


def get_rel_extrinsicM(ext_ref, ext_src):
    ''' Get the extrinisc matrix from ref_view to src_view '''
    return ext_src.dot( np.linalg.inv( ext_ref))
