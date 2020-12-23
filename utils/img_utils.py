import numpy as np
import math
import PIL.Image as image
import warping.homography as warp_homo
import copy

import torch
import torch.nn.functional as F
import torchvision
import cv2
import external.deval_lib.pyevaluatedepth_lib as dlib
epsilon = torch.finfo(float).eps

def eval_errors(errors):
    return dlib.evaluateErrors(errors)

def depth_error(predicted, truth):
    predicted_copy = predicted.copy()
    truth_copy = truth.copy()
    predicted_copy[predicted_copy == 0] = -1
    truth_copy[truth_copy == 0] = -1
    return dlib.depthError(predicted_copy + epsilon, truth_copy + epsilon)

def gaussian_torch(x, mu, sig, pow=2.):
    return torch.exp(-torch.pow(torch.abs(x - mu), pow) / (2 * torch.pow(sig, pow)))

def gaussian(x, mu, sig, pow=2.):
    return np.exp(-np.power(np.abs(x - mu), pow) / (2 * np.power(sig, pow)))

d_candi_expanded_d = dict()
def gen_soft_label_torch(d_candi, depthmap, variance, zero_invalid=False, pow=2.):
    global d_candi_expanded_d
    sstring = str(len(d_candi)) + "_" + str(depthmap.shape) + "_" + str(depthmap.device)
    if sstring not in d_candi_expanded_d.keys():
        d_candi_expanded = torch.tensor(d_candi).float().to(depthmap.device).unsqueeze(-1).unsqueeze(-1).repeat(1, depthmap.shape[0],
                                                                                                   depthmap.shape[1])
        d_candi_expanded_d[sstring] = d_candi_expanded
    else:
        d_candi_expanded = d_candi_expanded_d[sstring]

    # Warning, if a value in depthmap doesnt lie within d_candi range, it will become nan. zero_invalid forces it to -1
    sigma = torch.sqrt(variance)
    dists = gaussian_torch(d_candi_expanded, depthmap, sigma, pow)
    dists = dists/torch.sum(dists, dim=0)
    if zero_invalid: dists[dists != dists] = -1

    return dists

def gen_uniform(d_candi, depthmap):
    return torch.ones((len(d_candi), depthmap.shape[0], depthmap.shape[1])).to(depthmap.device) / len(d_candi)

def dpv_to_depthmap(dpv, d_candi, BV_log=False):
    if dpv.shape[0] != 1:
        raise Exception('Unable to handle this case')

    z = dpv.squeeze(0)
    if BV_log: z = torch.exp(z)
    d_candi_expanded = torch.tensor(d_candi).unsqueeze(1).unsqueeze(1).float().to(z.device)
    mean = torch.sum(d_candi_expanded * z, dim=0).unsqueeze(0)

    return mean

def demean(input):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    output = input.detach().clone()
    output[0, :, :] = input[0, :, :] * __imagenet_stats["std"][0] + __imagenet_stats["mean"][0]
    output[1, :, :] = input[1, :, :] * __imagenet_stats["std"][1] + __imagenet_stats["mean"][1]
    output[2, :, :] = input[2, :, :] * __imagenet_stats["std"][2] + __imagenet_stats["mean"][2]
    return output

def torchrgb_to_cv2(input, demean=True):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    input = input.detach().clone()
    if demean:
        input[0, :, :] = input[0, :, :] * __imagenet_stats["std"][0] + __imagenet_stats["mean"][0]
        input[1, :, :] = input[1, :, :] * __imagenet_stats["std"][1] + __imagenet_stats["mean"][1]
        input[2, :, :] = input[2, :, :] * __imagenet_stats["std"][2] + __imagenet_stats["mean"][2]
    return cv2.cvtColor(input[:, :, :].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)

def powerf(d_min, d_max, nDepth, power):
    f = lambda x: d_min + (d_max - d_min) * x
    x = np.linspace(start=0, stop=1, num=nDepth)
    x = np.power(x, power)
    candi = [f(v) for v in x]
    return np.array(candi)

def minpool(tensor, scale, default=0):
    if default:
        tensor_copy = tensor.clone()
        tensor_copy[tensor_copy == 0] = default
        tensor_small = -F.max_pool2d(-tensor_copy, scale)
        tensor_small[tensor_small == default] = 0
    else:
        tensor_small = -F.max_pool2d(-tensor, scale)
    return tensor_small

def intr_scale(intr, raw_img_size, img_size):
    uchange = float(img_size[0]) / float(raw_img_size[0])
    vchange = float(img_size[1]) / float(raw_img_size[1])
    intr_small = intr.copy()
    intr_small[0, :] *= uchange
    intr_small[1, :] *= vchange
    return intr_small

def intr_scale_unit(intr, scale=1.):
    intr_small = intr.copy()
    intr_small[0, :] *= scale
    intr_small[1, :] *= scale
    return intr_small

def depth_to_pts(depthf, intr):
    if depthf.shape[0] != 1:
        raise Exception('Unable to handle this case')

    depth = depthf[0,:,:]

    # Extract Params
    fx = intr[0,0]
    cx = intr[0,2]
    fy = intr[1,1]
    cy = intr[1,2]

    # Faster
    yfield, xfield = torch.meshgrid([torch.arange(0, depth.shape[0]).float().to(depthf.device),
                                     torch.arange(0, depth.shape[1]).float().to(depthf.device)])
    yfield = (yfield - cy) / fy
    xfield = (xfield - cx) / fx

    # Multiply
    X = torch.mul(xfield, depth)
    Y = torch.mul(yfield, depth)
    Z = depth
    ptcloud = torch.cat([X.unsqueeze(0),Y.unsqueeze(0),Z.unsqueeze(0)], 0)

    return ptcloud

def hack(cloud):
    fcloud = np.zeros(cloud.shape).astype(np.float32)
    for i in range(0, cloud.shape[0]):
        fcloud[i] = cloud[i]
    return fcloud

def tocloud(depth, rgb, intr, extr=None, rgbr=None):
    pts = depth_to_pts(depth, intr)
    pts = pts.reshape((3, pts.shape[1] * pts.shape[2]))
    # pts_numpy = pts.numpy()

    # Attempt to transform
    pts = torch.cat([pts, torch.ones((1, pts.shape[1])).to(depth.device)])
    if extr is not None:
        transform = torch.inverse(extr)
        pts = torch.matmul(transform, pts)
    pts_numpy = pts[0:3, :].cpu().numpy()

    # Convert Color
    pts_color = (rgb.reshape((3, rgb.shape[1] * rgb.shape[2])) * 255).cpu().numpy()
    pts_normal = np.zeros((3, rgb.shape[1] * rgb.shape[2]))

    # RGBR
    if rgbr is not None:
        pts_color[0, :] = rgbr[0]
        pts_color[1, :] = rgbr[1]
        pts_color[2, :] = rgbr[2]

    # Visualize
    all_together = np.concatenate([pts_numpy, pts_color, pts_normal], 0).astype(np.float32).T
    all_together = hack(all_together)
    return all_together

def convert_flowfield(flowfield):
    yv, xv = torch.meshgrid([torch.arange(0, flowfield.shape[1]).float().to(flowfield.device), torch.arange(0, flowfield.shape[2]).float().to(flowfield.device)])
    ystep = 2. / float(flowfield.shape[1] - 1)
    xstep = 2. / float(flowfield.shape[2] - 1)
    flowfield[0, :, :, 0] = -1 + xv * xstep - flowfield[0, :, :, 0] * xstep
    flowfield[0, :, :, 1] = -1 + yv * ystep - flowfield[0, :, :, 1] * ystep
    return flowfield

def compute_unc_field(dpv_refined_predicted, dpv_refined_truth, d_candi, intr_refined, mask_refined, cfg):
    unc_field_truth, _ = gen_ufield(dpv_refined_truth, d_candi, intr_refined.squeeze(0), BV_log=False, mask=mask_refined, cfg=cfg)
    unc_field_predicted, debugmap = gen_ufield(dpv_refined_predicted, d_candi, intr_refined.squeeze(0), BV_log=True, cfg=cfg)
    return unc_field_truth, unc_field_predicted, debugmap

def compute_unc_rmse(unc_field_truth, unc_field_predicted, d_candi, plot=False):
    # Get Depth comparison
    unc_field_truth_depth = dpv_to_depthmap(unc_field_truth.unsqueeze(2), d_candi, BV_log=False).squeeze(0).squeeze(0)
    unc_field_predicted_depth = dpv_to_depthmap(unc_field_predicted.unsqueeze(2), d_candi, BV_log=False).squeeze(0).squeeze(0)
    unc_field_predicted_depth[0] = 0
    unc_field_predicted_depth[-1] = 0
    unc_field_mask = ~torch.isnan(unc_field_truth_depth) & ~torch.isnan(unc_field_predicted_depth)
    unc_field_truth_depth[~unc_field_mask] = 0.
    unc_field_predicted_depth[~unc_field_mask] = 0.
    unc_field_rmse = torch.sqrt(torch.sum((unc_field_truth_depth*unc_field_mask - unc_field_predicted_depth*unc_field_mask)**2)/torch.sum(unc_field_mask))
    unc_field_rmse = torch.sum(torch.abs(unc_field_truth_depth*unc_field_mask - unc_field_predicted_depth*unc_field_mask))/torch.sum(unc_field_mask)
    if plot:
        import matplotlib.pyplot as plt
        # Plot
        plt.ion()
        plt.cla()
        plt.plot((unc_field_truth_depth*unc_field_mask).cpu().numpy())
        plt.plot((unc_field_predicted_depth*unc_field_mask).cpu().numpy())
        plt.pause(0.05)
    return unc_field_rmse

spread_kernel = None
spread_conv = None
def spread_dpv_hack(dpv, N=5):
    # torch.Size([128, 384])
    # torch.Size([1, 1, 5, 5])
    global spread_kernel
    global spread_conv
    dpv_permuted = dpv.permute(0, 3, 2, 1)

    if spread_kernel is None:
        kernel = torch.Tensor(np.zeros((N, N)).astype(np.float32))
        kernel[int(N / 2), :] = 1.
        # kernel[2,2] = 1.
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat((1, 1, 1, 1))
        kernel_dict = {'weight': kernel.to(dpv_permuted.device), 'padding': N // 2}
        spread_kernel = kernel_dict.copy()
        spread_conv = torch.nn.Conv2d(in_channels=dpv_permuted.shape[1], out_channels=dpv_permuted.shape[1], groups=dpv_permuted.shape[1], kernel_size= (N,N), padding=N//2, bias=False).cuda()
        spread_conv.weight.requires_grad = False
        spread_conv.weight[:,:,:,:] = kernel

    dpv_permuted = spread_conv(dpv_permuted)

    dpv = dpv_permuted.permute(0, 3, 2, 1)
    tofuse_dpv = dpv / torch.sum(dpv, dim=1).unsqueeze(1)
    return tofuse_dpv

def upsample_dpv(dpv_refined_predicted, N=64, BV_log=False):
    if dpv_refined_predicted.shape[1] == N:
        return dpv_refined_predicted
    if BV_log:
        dpv_refined_predicted = torch.exp(dpv_refined_predicted)
    dpv_refined_predicted = dpv_refined_predicted.permute(0,2,1,3)
    dpv_refined_predicted = F.upsample(dpv_refined_predicted, size=[N,dpv_refined_predicted.shape[3]], mode='bilinear')
    dpv_refined_predicted = dpv_refined_predicted.permute(0,2,1,3)
    dpv_refined_predicted = dpv_refined_predicted / torch.sum(dpv_refined_predicted, dim=1).unsqueeze(1)
    if BV_log:
        dpv_refined_predicted = torch.log(dpv_refined_predicted)
    return dpv_refined_predicted

def cull_depth(depth, intr_up, pshift=5):
    depth = depth.unsqueeze(0)

    # Generate Shiftmap
    flowfield = torch.zeros((1, depth.shape[2], depth.shape[3], 2)).float().to(depth.device)
    flowfield_inv = torch.zeros((1, depth.shape[2], depth.shape[3], 2)).float().to(depth.device)
    flowfield[:, :, :, 1] = pshift
    flowfield_inv[:, :, :, 1] = -pshift
    convert_flowfield(flowfield)
    convert_flowfield(flowfield_inv)

    # Shift the DPV
    depth_shifted = F.grid_sample(depth, flowfield, mode='nearest').squeeze(0)

    pts_shifted = depth_to_pts(depth_shifted, intr_up)
    zero_mask = (~((pts_shifted[1, :, :] > 1.0) | (pts_shifted[1, :, :] < 0.6) | (pts_shifted[2, :, :] > 100-1) | (pts_shifted[2, :, :] < 3))).float() # THEY ALL SEEM TO BE DIFF HEIGHT? (CHECK CALIB)
    #zero_mask = zero_mask*0 + 1

    depth_shifted = depth_shifted * zero_mask

    depth_reverted = F.grid_sample(depth_shifted.unsqueeze(0), flowfield_inv, mode='nearest').squeeze(0)

    return depth_reverted

def gen_ufield(dpv_predicted, d_candi, intr_up, visualizer=None, img=None, BV_log=True, normalize=False, mask=None, cfg=None, cfgx=None):
    if cfgx is not None:
        pshift = cfgx["unc_ang"]
        zstart = cfgx["unc_shift"]
        zend = zstart + cfgx["unc_span"]
        maxd = 100.
        mind = 3.
        quash_limit = True
    else:
        if "kitti" in cfg.data.dataset_path:
            pshift = 5
            zstart = 0.6
            zend = zstart + 0.3
            maxd = 100.
            mind = 0.
            quash_limit = False
        elif "ilim" in cfg.data.dataset_path:
            pshift = 0
            zstart = 1.0
            zend = zstart + 0.3
            maxd = 100.
            mind = 3.
            quash_limit = True

    # Generate Shiftmap
    if pshift != 0:
        flowfield = torch.zeros((1, dpv_predicted.shape[2], dpv_predicted.shape[3], 2)).float().to(dpv_predicted.device)
        flowfield_inv = torch.zeros((1, dpv_predicted.shape[2], dpv_predicted.shape[3], 2)).float().to(dpv_predicted.device)
        flowfield[:, :, :, 1] = pshift
        flowfield_inv[:, :, :, 1] = -pshift
        convert_flowfield(flowfield)
        convert_flowfield(flowfield_inv)

        # Shift the DPV
        dpv_shifted = F.grid_sample(dpv_predicted, flowfield, mode='nearest')
    else:
        dpv_shifted = dpv_predicted.clone()

    # To Depthmap
    depthmap_shifted = dpv_to_depthmap(dpv_shifted, d_candi, BV_log=BV_log)
    depthmap_predicted = dpv_to_depthmap(dpv_predicted, d_candi, BV_log=BV_log)

    # Get Mask for Pts within Y range
    # This is bad as i dont want zero regions. so i max it out now
    pts_shifted = depth_to_pts(depthmap_shifted, intr_up)
    #zero_mask = (~((pts_shifted[1,:,:] > 1.4) | (pts_shifted[1,:,:] < -1.0))).float()
    #zero_mask = (~((pts_shifted[1, :, :] > 1.3) | (pts_shifted[1, :, :] < 1.0))).float()
    # (~((pts_shifted[1, :, :] > 1.0) | (pts_shifted[1, :, :] < 0.5)
    zero_mask = (~((pts_shifted[1, :, :] > zend) | (pts_shifted[1, :, :] < zstart) | (pts_shifted[2, :, :] > maxd-1) | (pts_shifted[2, :, :] < mind))).float() # THEY ALL SEEM TO BE DIFF HEIGHT? (CHECK CALIB)
    if mask is not None:
        if pshift != 0:
            mask_shifted = F.grid_sample(mask.unsqueeze(1), flowfield, mode='nearest').squeeze(1)
        else:
            mask_shifted = mask.clone()
        zero_mask = zero_mask * mask_shifted.squeeze(0)

    # Need an algorithm to ensure two highly varied depthms are not in same column?
    if quash_limit:
        quash_range = 1.
        depthmap_shifted_cleaned = (depthmap_shifted * zero_mask).squeeze(0) # 256x320
        depthmap_shifted_cleaned[depthmap_shifted_cleaned == 0] = 1000
        min_along_column, _ = torch.min(depthmap_shifted_cleaned, axis=0) # 320
        quash_mask = ((depthmap_shifted_cleaned > min_along_column-quash_range) & (depthmap_shifted_cleaned < min_along_column+quash_range))
        quash_mask = quash_mask.float()
        zero_mask = zero_mask * quash_mask

    # Shift Mask
    if pshift != 0:
        zero_mask_predicted = F.grid_sample(zero_mask.unsqueeze(0).unsqueeze(0), flowfield_inv, mode='nearest').squeeze(0).squeeze(0)
    else:
        zero_mask_predicted = zero_mask.clone()
    depthmap_predicted_zero = depthmap_predicted * zero_mask_predicted

    # DPV Zero out and collapse
    zero_mask_predicted = zero_mask_predicted.repeat([len(d_candi), 1, 1], 0, 1).unsqueeze(0)
    if BV_log:
        dpv_plane = torch.sum(torch.exp(dpv_predicted) * zero_mask_predicted, axis = 2) # [1,64,384]
    else:
        dpv_plane = torch.sum(dpv_predicted * zero_mask_predicted, axis=2)  # [1,64,384]

    # Normalize
    ax = torch.sum(zero_mask, axis=0)
    dpv_plane = dpv_plane / ax
    #dpv_plane = F.softmax(dpv_plane, dim=1)

    # Make 0 to 1 for visualization
    minval, _ = dpv_plane.min(1) # [1,384]
    maxval, _ = dpv_plane.max(1)  # [1,384]
    if(normalize): dpv_plane = (dpv_plane - minval) / (maxval - minval)

    return dpv_plane, depthmap_predicted_zero

def gen_dpv_withmask(dmaps, masks, d_candi, var=0.3):
    # torch.Size([2, 64, 96])
    # torch.Size([2, 1, 64, 96])
    tofuse_dpv = []
    truth_var = torch.tensor(var)
    for b in range(0, dmaps.shape[0]):
        dmap = dmaps[b, :, :]
        mask = masks[b, 0, :, :].unsqueeze(0)
        mask_inv = 1. - mask
        truth_dpv = gen_soft_label_torch(d_candi, dmap, truth_var, zero_invalid=True)
        uni_dpv = gen_uniform(d_candi, dmap)
        modified_dpv = truth_dpv * mask + uni_dpv * mask_inv
        tofuse_dpv.append(modified_dpv.unsqueeze(0))
    tofuse_dpv = torch.cat(tofuse_dpv)
    tofuse_dpv = torch.clamp(tofuse_dpv, epsilon, 1.)
    return tofuse_dpv

def unitQ_to_quat( unitQ, quat):
    '''
    Unit quaternion (xyz parameterization) to quaternion
    unitQ - 3 vector,
    quat - 4 vector, TUM format quaternion [x y z w]
    '''
    x, y, z = unitQ[0], unitQ[1], unitQ[2]
    alpha2 = x**2 + y**2 + z**2

    quat[3] = 2* x / (alpha2 + 1)
    quat[0] = 2* y / (alpha2 + 1)
    quat[1] = 2* z / (alpha2 + 1)
    quat[2] = (1-alpha2) / (1+ alpha2)

def unitQ_to_quat_inv( unitQ, quat):
    '''
    Unit quaternion (xyz parameterization) to inverse quaternion
    unitQ - 3 vector,
    quat - 4 vector, TUM format quaternion [x y z w]
    '''
    x, y, z = unitQ[0], unitQ[1], unitQ[2]
    alpha2 = x**2 + y**2 + z**2

    quat[0] = -2* y / (alpha2 + 1)
    quat[1] = -2* z / (alpha2 + 1)
    quat[2] = -(1-alpha2) / (1+ alpha2)
    quat[3] = 2* x / (alpha2 + 1)

def quat_to_unitQ(quat, unitQ):
    '''
    get Unit quaternion (xyz parameterization) from quaternion
    quat - 4 vector, TUM format quaternion [x y z w]
    unitQ - 3 vector,
    '''
    q1,q2,q3,q0 = quat[0], quat[1], quat[2], quat[3]
    alpha2 = (1-q3) / (1+q3)

    x = q0*(alpha2+1) * .5
    y = q1*(alpha2+1) * .5
    z = q2*(alpha2+1) * .5

    unitQ[0] = x
    unitQ[1] = y
    unitQ[2] = z

def quaternion_to_rotation(q, is_tensor = False, R_tensor=None, TUM_format=True):
    '''
    input:
    q - 4 element np array:
    q in the TUM monoVO format: qx qy qz qw

    is_tensor - if use tensor array

    R_tensor - 3x3 tensor, should be initialized

    output:
    Rot - 3x3 rotation matrix
    '''
    if is_tensor and R_tensor is not None:
        Rot = R_tensor
    elif is_tensor and R_tensor is None:
        Rot = torch.zeros(3,3).to(q.device)
    else:
        Rot = np.zeros((3,3))

    if TUM_format:
        w, x, y, z = q[3], q[0], q[1], q[2]
    else:
        w, x, y, z =  q[0], q[1], q[2], q[3]

    s = 1 / (w**2 + x**2 + y**2 + z**2)
    Rot[0, 0] = 1 - 2 * s * ( y**2 + z**2 )
    Rot[1, 1] = 1 - 2 * s * ( x**2 + z**2)
    Rot[2, 2] = 1 - 2 * s * ( x**2 + y**2)

    Rot[0, 1] = 2* ( x*y - w * z)
    Rot[1, 0] = 2* ( x*y + w * z)

    Rot[0, 2] = 2* ( x*z + w * y)
    Rot[2, 0] = 2* ( x*z - w * y)

    Rot[1,2] = 2* ( y*z - w * x)
    Rot[2,1] = 2* ( y*z + w * x)

    return Rot

def rotation_to_quaternion(R, quat):
    '''
    reference: http://www.engr.ucr.edu/~farrell/AidedNavigation/D_App_Quaternions/Rot2Quat.pdf
    NOTE: quat is in TUM format: quat = [qx qy qz qw]

    ref:
    https://engineering.purdue.edu/CE/Academics/Groups/Geomatics/DPRG/Slides/Chapter7_Quaternion
    '''

    assert quat.dim() == 1 and len(quat) == 4, 'quat should be of right shape !'

    if R[0, 0] + R[1, 1] + R[2, 2] + 1 > 0:
        quat[3] = .5 * math.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1)
        s = 1 / 4 / quat[3]
        quat[0] = s * (R[2, 1] - R[1, 2])
        quat[1] = s * (R[0, 2] - R[2, 0])
        quat[2] = s * (R[1, 0] - R[0, 1])

    elif R[0, 0] - R[1, 1] - R[2, 2] + 1 > 0:
        quat[0] = .5 * math.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1)
        s = 1 / 4 / quat[0]
        quat[1] = s * (R[1, 0] + R[0, 1])
        quat[2] = s * (R[0, 2] + R[2, 0])
        quat[3] = s * (R[2, 1] + R[1, 2])

    elif R[1, 1] - R[0, 0] - R[2, 2] + 1 > 0:
        quat[1] = .5 * math.sqrt(R[1, 1] - R[0, 0] - R[2, 2] + 1)
        s = 1 / 4 / quat[0]
        quat[0] = s * (R[1, 0] + R[0, 1])
        quat[2] = s * (R[1, 2] + R[2, 1])
        quat[3] = s * (R[0, 2] - R[2, 0])

    elif R[2, 2] - R[0, 0] - R[1, 1] + 1 > 0:
        quat[2] = .5 * math.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1)
        s = 1 / 4 / quat[0]
        quat[0] = s * (R[2, 0] + R[0, 2])
        quat[1] = s * (R[2, 1] + R[1, 2])
        quat[3] = s * (R[1, 0] - R[0, 1])


def unitquat_to_rotation(r_uq):
    assert isinstance(r_uq, torch.Tensor)
    r_q = torch.zeros(4).to(r_uq.device)
    unitQ_to_quat(r_uq, r_q)
    R = quaternion_to_rotation(r_q, is_tensor=True)
    return R

def rotation_to_unitquat(R):
    r_q = torch.zeros(4).to(R.device)
    r_uq = torch.zeros(3).to(R.device)
    rotation_to_quaternion(R, r_q)
    quat_to_unitQ( r_q, r_uq)
    return r_uq

def add_noise2pose(src_cam_poses_in, noise_level =.2):
    '''
    noise_level - gaussian_sigma / norm_r r, gaussian_sigma/ norm_t for t
    add Gaussian noise to the poses:
    for R: add in the unit-quaternion space
    for t: add in the raw space
    '''

    src_cam_poses_out = torch.zeros( src_cam_poses_in.shape)
    src_cam_poses_out[:, :, 3, 3] = 1.
    # for each batch #
    for ibatch in range(src_cam_poses_in.shape[0]):
        src_cam_poses_perbatch = src_cam_poses_in[ibatch, ...]
        for icam in range(src_cam_poses_perbatch.shape[0]):
            src_cam_pose = src_cam_poses_perbatch[icam, ...]

            # convert to unit quaternion #
            r = rotation_to_unitquat(src_cam_pose[:3, :3].to(src_cam_pose.device))
            t = src_cam_pose[:3, 3]

            # add noise to r and t #
            sigma_r = noise_level * r.norm()
            sigma_t = noise_level * t.norm()
            r = r + torch.randn(r.shape).to(src_cam_pose.device) * sigma_r
            t = t + torch.randn(t.shape) * sigma_t

            # put back in to src_cam_poses_out #
            src_cam_poses_out[ibatch, icam, :3, :3] = unitquat_to_rotation(r)
            src_cam_poses_out[ibatch, icam, :3, 3] = t

    return src_cam_poses_out