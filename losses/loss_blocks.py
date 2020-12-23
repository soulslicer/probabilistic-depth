import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.img_utils as img_utils
from utils import inverse_warp as iv

# Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
def TernaryLoss(im, im_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)

    return dist * mask


def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist

def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / mask.sum()
    return mean_value

def edge_aware_smoothness_loss(pred_disp, img, max_scales):
    def gradient_x(img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(img):
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy

    def get_edge_smoothness(img, pred):
        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x),
                                          1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y),
                                          1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.

    s = 0
    for scaled_disp in pred_disp:
        s += 1
        if s > max_scales:
            break

        b, _, h, w = scaled_disp.size()
        scaled_img = F.adaptive_avg_pool2d(img, (h, w))
        loss += get_edge_smoothness(scaled_img, scaled_disp) * weight
        weight /= 4.0

    return loss

def rgb_stereo_consistency_loss(src_rgb_img, target_rgb_img, target_depth_map, pose_target2src, intr, viz=False):
    # Warp
    target_warped_rgb_img, valid_points = iv.inverse_warp(src_rgb_img, target_depth_map, pose_target2src, intr)

    # Mask (Should just be doing top half and valid_points)
    tophalf = torch.ones(valid_points.shape).bool().to(src_rgb_img.device);
    tophalf[:, 0:int(tophalf.shape[1] / 3), :] = False
    full_mask = valid_points & tophalf
    full_mask = full_mask.float()
    target_rgb_img = target_rgb_img * full_mask
    target_warped_rgb_img = target_warped_rgb_img * full_mask

    # Integrate loss here too and visualize for changing pose
    diff_img = (target_rgb_img - target_warped_rgb_img).abs()  # [1, 3, 256, 384]
    # ssim_map = (0.5 * (1 - ssim(target_rgb_img, target_warped_rgb_img))).clamp(0, 1)
    # diff_img = (0.15 * diff_img + 0.85 * ssim_map)
    photo_error = mean_on_mask(diff_img, full_mask)

    # Visualize
    if viz:
        import cv2
        import numpy as np
        img_color_target = img_utils.torchrgb_to_cv2(target_rgb_img.squeeze(0))
        img_color_diff = img_utils.torchrgb_to_cv2(diff_img.squeeze(0), False)
        img_depth_target = cv2.cvtColor(target_depth_map[0, :, :].detach().cpu().numpy() / 100., cv2.COLOR_GRAY2BGR)
        img_color_warped_target = img_utils.torchrgb_to_cv2(target_warped_rgb_img.squeeze(0))
        diff = np.abs(img_color_warped_target - img_color_target)
        combined = np.hstack([img_color_target, img_color_warped_target, diff, img_depth_target])
        cv2.imshow("win", combined)
        cv2.waitKey(15)

    return photo_error

def depth_stereo_consistency_loss(src_depth_img, target_depth_img, src_depth_mask, target_depth_mask, pose_target2src, intr):
    # Transform (Below needed only if baseline z changes or big trans)
    src_depth_img_trans = iv.transform_dmap(src_depth_img[0, 0, :, :], torch.inverse(pose_target2src), intr[0, :, :])
    src_depth_img_trans = (src_depth_img_trans.unsqueeze(0) * src_depth_mask.float()).unsqueeze(0)
    target_warped_depth_img, valid_points = iv.inverse_warp(src_depth_img_trans, target_depth_img.squeeze(0),
                                                            pose_target2src, intr, 'nearest')

    # Mask (Should just be doing top half and valid_points)
    warp_mask = target_warped_depth_img > 0.
    tophalf = torch.ones(valid_points.shape).bool().to(src_depth_img.device)
    tophalf[:, 0:int(tophalf.shape[1] / 3), :] = False
    full_mask = valid_points & tophalf & warp_mask  # We should not need target_depth_mask
    full_mask = full_mask.float()
    target_depth_img = target_depth_img * full_mask
    target_warped_depth_img = target_warped_depth_img * full_mask

    # Score
    target_depth_img = target_depth_img.clamp(min=1e-3)
    target_warped_depth_img = target_warped_depth_img.clamp(min=1e-3)
    diff_depth = ((target_depth_img - target_warped_depth_img).abs() /
                  (target_depth_img + target_warped_depth_img).abs()).clamp(0, 1)
    dc_loss = mean_on_mask(diff_depth, full_mask)
    # diff_depth = (target_depth_img - target_warped_depth_img).abs()
    # reconstruction_loss = mean_on_mask(diff_depth, full_mask)
    return dc_loss

def depth_consistency_loss(large_dm, small_dm):
    tophalf = torch.ones(small_dm.shape).bool().to(large_dm.device);
    tophalf[:, 0:int(tophalf.shape[1] / 3), :] = False
    #downscaled_dm = F.interpolate(large_dm.unsqueeze(0), size=[small_dm.shape[1], small_dm.shape[2]], mode='nearest').squeeze(0)
    #downscaled_dm = F.max_pool2d(large_dm.unsqueeze(0), 4).squeeze(0)
    downscaled_dm = img_utils.minpool(large_dm.unsqueeze(0), 4).squeeze(0)
    small_dm = small_dm.clamp(min=1e-3)
    downscaled_dm = downscaled_dm.clamp(min=1e-3)
    diff_depth = ((downscaled_dm - small_dm).abs() /
                  (downscaled_dm + small_dm).abs()).clamp(0, 1)
    dc_loss = mean_on_mask(diff_depth, tophalf.float())
    return dc_loss

def soft_cross_entropy_loss(soft_label, x, mask=None, BV_log=False):
    if BV_log:
        log_x_softmax = x
    else:
        x_softmax = F.softmax(x, dim=1)
        log_x_softmax = torch.log(x_softmax)

    loss = -torch.sum(soft_label * log_x_softmax, 1)

    if mask is not None:
        loss = loss * mask
        nonzerocount = (mask == 1).sum()
        if nonzerocount == 0: return 0.
        loss = torch.sum(loss)/nonzerocount
    else:
        loss = torch.mean(loss)
    return loss

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad_1st(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2

    return loss_x.mean() / 2. + loss_y.mean() / 2.
