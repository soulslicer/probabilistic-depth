import torch.nn as nn
import torch.nn.functional as F
import torch
import utils.img_utils as img_utils
from .loss_blocks import *


class BaseLoss(nn.modules.Module):
    def __init__(self, cfg, id):
        super(BaseLoss, self).__init__()
        self.cfg = cfg
        self.id = id

    def forward(self, output, target):
        output_left, output_right = output
        target_left, target_right = target

        # Extract Variables
        device = output_left["output"][-1].device
        d_candi = target_left["d_candi"]
        T_left2right = target_left["T_left2right"]
        BV_cur_left_array = output_left["output"]
        BV_cur_right_array = output_right["output"]
        BV_cur_refined_left_array = output_left["output_refined"]
        BV_cur_refined_right_array = output_right["output_refined"]
        gt_input_left = target_left
        gt_input_right = target_right

        # NLL Loss for Low Res
        ce_loss = 0
        ce_count = 0
        for ind in range(len(BV_cur_left_array)):
            BV_cur_left = BV_cur_left_array[ind]
            BV_cur_right = BV_cur_right_array[ind]
            for ibatch in range(BV_cur_left.shape[0]):
                ce_count += 1
                # Left Losses
                ce_loss = ce_loss + soft_cross_entropy_loss(
                    gt_input_left["soft_labels"][ibatch].unsqueeze(0),
                    BV_cur_left[ibatch, :, :, :].unsqueeze(0),
                    mask=gt_input_left["masks"][ibatch, :, :, :],
                    BV_log=True)
                # Right Losses
                ce_loss = ce_loss + soft_cross_entropy_loss(
                    gt_input_right["soft_labels"][ibatch].unsqueeze(0),
                    BV_cur_right[ibatch, :, :, :].unsqueeze(0),
                    mask=gt_input_right["masks"][ibatch, :, :, :],
                    BV_log=True)

        # NLL Loss for High Res
        for ind in range(len(BV_cur_refined_left_array)):
            BV_cur_refined_left = BV_cur_refined_left_array[ind]
            BV_cur_refined_right = BV_cur_refined_right_array[ind]
            for ibatch in range(BV_cur_refined_left.shape[0]):
                ce_count += 1
                # Left Losses
                ce_loss = ce_loss + soft_cross_entropy_loss(
                    gt_input_left["soft_labels_imgsize"][ibatch].unsqueeze(0),
                    BV_cur_refined_left[ibatch, :, :, :].unsqueeze(0),
                    mask=gt_input_left["masks_imgsizes"][ibatch, :, :, :],
                    BV_log=True)
                # Right Losses
                ce_loss = ce_loss + soft_cross_entropy_loss(
                    gt_input_right["soft_labels_imgsize"][ibatch].unsqueeze(0),
                    BV_cur_refined_right[ibatch, :, :, :].unsqueeze(0),
                    mask=gt_input_right["masks_imgsizes"][ibatch, :, :, :],
                    BV_log=True)

        # Get Last BV_cur
        BV_cur_left = BV_cur_left_array[-1]
        BV_cur_right = BV_cur_right_array[-1]
        BV_cur_refined_left = BV_cur_refined_left_array[-1]
        BV_cur_refined_right = BV_cur_refined_right_array[-1]

        # Regress all depthmaps once here
        small_dm_left_arr = []
        large_dm_left_arr = []
        small_dm_right_arr = []
        large_dm_right_arr = []
        for ibatch in range(BV_cur_left.shape[0]):
            small_dm_left_arr.append(
                img_utils.dpv_to_depthmap(BV_cur_left[ibatch, :, :, :].unsqueeze(0), d_candi, BV_log=True))
            large_dm_left_arr.append(
                img_utils.dpv_to_depthmap(BV_cur_refined_left[ibatch, :, :, :].unsqueeze(0), d_candi, BV_log=True))
            small_dm_right_arr.append(
                img_utils.dpv_to_depthmap(BV_cur_right[ibatch, :, :, :].unsqueeze(0), d_candi, BV_log=True))
            large_dm_right_arr.append(
                img_utils.dpv_to_depthmap(BV_cur_refined_right[ibatch, :, :, :].unsqueeze(0), d_candi, BV_log=True))

        # Downsample Consistency Loss (Should we even have a mask here?)
        dc_loss = 0
        for ibatch in range(BV_cur_left.shape[0]):
            if self.cfg.loss.dc_mul == 0: break
            # Left
            mask_left = gt_input_left["masks"][ibatch, :, :, :]
            small_dm_left = small_dm_left_arr[ibatch]
            large_dm_left = large_dm_left_arr[ibatch]
            dc_loss = dc_loss + depth_consistency_loss(large_dm_left, small_dm_left)
            # Right
            mask_right = gt_input_right["masks"][ibatch, :, :, :]
            small_dm_right = small_dm_right_arr[ibatch]
            large_dm_right = large_dm_right_arr[ibatch]
            dc_loss = dc_loss + depth_consistency_loss(large_dm_right, small_dm_right)

        # Depth Stereo Consistency Loss
        pose_target2src = T_left2right
        pose_target2src = torch.unsqueeze(pose_target2src, 0).to(device)
        pose_src2target = torch.inverse(T_left2right)
        pose_src2target = torch.unsqueeze(pose_src2target, 0).to(device)
        dsc_loss = 0
        for ibatch in range(BV_cur_left.shape[0]):
            if self.cfg.loss.dsc_mul == 0: break
            # Get all Data
            intr_up_left = gt_input_left["intrinsics_up"][ibatch, :, :].unsqueeze(0)
            intr_left = gt_input_left["intrinsics"][ibatch, :, :].unsqueeze(0)
            intr_up_right = gt_input_right["intrinsics_up"][ibatch, :, :].unsqueeze(0)
            intr_right = gt_input_right["intrinsics"][ibatch, :, :].unsqueeze(0)
            depth_up_left = large_dm_left_arr[ibatch].unsqueeze(0)
            depth_left = small_dm_left_arr[ibatch].unsqueeze(0)
            depth_up_right = large_dm_right_arr[ibatch].unsqueeze(0)
            depth_right = small_dm_right_arr[ibatch].unsqueeze(0)
            mask_up_left = gt_input_left["masks_imgsizes"][ibatch, :, :, :]
            mask_left = gt_input_left["masks"][ibatch, :, :, :]
            mask_up_right = gt_input_right["masks_imgsizes"][ibatch, :, :, :]
            mask_right = gt_input_right["masks"][ibatch, :, :, :]
            # Right to Left
            dsc_loss = dsc_loss + depth_stereo_consistency_loss(depth_up_right, depth_up_left, mask_up_right,
                                                                mask_up_left,
                                                                pose_target2src, intr_up_left)
            dsc_loss = dsc_loss + depth_stereo_consistency_loss(depth_right, depth_left, mask_right, mask_left,
                                                                pose_target2src, intr_left)
            # Left to Right
            dsc_loss = dsc_loss + depth_stereo_consistency_loss(depth_up_left, depth_up_right, mask_up_left,
                                                                mask_up_right,
                                                                pose_src2target, intr_up_right)
            dsc_loss = dsc_loss + depth_stereo_consistency_loss(depth_left, depth_right, mask_left, mask_right,
                                                                pose_src2target, intr_right)

        # RGB Stereo Consistency Loss (Just on high res)
        rsc_loss = 0
        for ibatch in range(BV_cur_left.shape[0]):
            if self.cfg.loss.rsc_mul == 0: break
            intr_up_left = gt_input_left["intrinsics_up"][ibatch, :, :].unsqueeze(0)
            intr_up_right = gt_input_right["intrinsics_up"][ibatch, :, :].unsqueeze(0)
            depth_up_left = large_dm_left_arr[ibatch]
            depth_up_right = large_dm_right_arr[ibatch]
            rgb_up_left = gt_input_left["rgb"][ibatch, -1, :, :, :].unsqueeze(0)
            rgb_up_right = gt_input_right["rgb"][ibatch, -1, :, :, :].unsqueeze(0)
            mask_up_left = gt_input_left["masks_imgsizes"][ibatch, :, :, :]
            mask_up_right = gt_input_right["masks_imgsizes"][ibatch, :, :, :]
            # Right to Left
            # src_rgb_img, target_rgb_img, target_depth_map, pose_target2src, intr
            rsc_loss = rsc_loss + rgb_stereo_consistency_loss(rgb_up_right, rgb_up_left, depth_up_left,
                                                              pose_target2src,
                                                              intr_up_left)
            # Left to Right
            rsc_loss = rsc_loss + rgb_stereo_consistency_loss(rgb_up_left, rgb_up_right, depth_up_right,
                                                              pose_src2target,
                                                              intr_up_right)

        # RGB Stereo Consistency Loss (Low res)
        rsc_low_loss = 0
        for ibatch in range(BV_cur_left.shape[0]):
            if self.cfg.loss.rsc_low_mul == 0: break
            intr_left = gt_input_left["intrinsics"][ibatch, :, :].unsqueeze(0)
            intr_right = gt_input_right["intrinsics"][ibatch, :, :].unsqueeze(0)
            depth_left = small_dm_left_arr[ibatch]
            depth_right = small_dm_right_arr[ibatch]
            rgb_left = F.interpolate(gt_input_left["rgb"][ibatch, -1, :, :, :].unsqueeze(0), scale_factor=0.25,
                                     mode='bilinear')
            rgb_right = F.interpolate(gt_input_right["rgb"][ibatch, -1, :, :, :].unsqueeze(0), scale_factor=0.25,
                                      mode='bilinear')
            # Right to Left
            # src_rgb_img, target_rgb_img, target_depth_map, pose_target2src, intr
            rsc_low_loss = rsc_low_loss + rgb_stereo_consistency_loss(rgb_right, rgb_left, depth_left,
                                                                      pose_target2src,
                                                                      intr_left)
            # Left to Right
            rsc_low_loss = rsc_low_loss + rgb_stereo_consistency_loss(rgb_left, rgb_right, depth_right,
                                                                      pose_src2target,
                                                                      intr_right)

        # Smoothness loss (Just on high res)
        smooth_loss = 0
        for ibatch in range(BV_cur_left.shape[0]):
            if self.cfg.loss.smooth_mul == 0: break
            depth_up_left = large_dm_left_arr[ibatch].unsqueeze(0)
            depth_up_right = large_dm_right_arr[ibatch].unsqueeze(0)
            rgb_up_left = gt_input_left["rgb"][ibatch, -1, :, :, :].unsqueeze(0)
            rgb_up_right = gt_input_right["rgb"][ibatch, -1, :, :, :].unsqueeze(0)
            # Left
            smooth_loss = smooth_loss + edge_aware_smoothness_loss([depth_up_left], rgb_up_left, 1)
            # Right
            smooth_loss = smooth_loss + edge_aware_smoothness_loss([depth_up_right], rgb_up_right, 1)

        # All Loss
        loss = torch.tensor(0.).to(device)

        # Depth Losses
        bsize = torch.tensor(float(BV_cur_left.shape[0] * 2)).to(device)
        if bsize != 0:
            ce_loss = (ce_loss / ce_count) * self.cfg.loss.ce_mul
            dsc_loss = (dsc_loss / bsize) * self.cfg.loss.dsc_mul
            dc_loss = (dc_loss / bsize) * self.cfg.loss.dc_mul
            rsc_loss = (rsc_loss / bsize) * self.cfg.loss.rsc_mul
            rsc_low_loss = (rsc_low_loss / bsize) * self.cfg.loss.rsc_low_mul
            smooth_loss = (smooth_loss / bsize) * self.cfg.loss.smooth_mul
            loss += (ce_loss + dsc_loss + dc_loss + rsc_loss + rsc_low_loss + smooth_loss)

        return loss

class DefaultLoss(nn.modules.Module):
    def __init__(self, cfg, id):
        super(DefaultLoss, self).__init__()
        self.cfg = cfg
        self.id = id

    def forward(self, output, target):
        """
        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        output_left, output_right = output
        target_left, target_right = target

        left_loss = 0.
        right_loss = 0.
        for b in range(0, len(target_left["soft_labels"])):
            label_left = target_left["soft_labels"][b].unsqueeze(0)
            label_right = target_right["soft_labels"][b].unsqueeze(0)

            left_loss += torch.sum(torch.abs(output_left["output"][-1] - 0))
            right_loss += torch.sum(torch.abs(output_right["output"][-1] - 0))

        loss = left_loss + right_loss

        return loss
