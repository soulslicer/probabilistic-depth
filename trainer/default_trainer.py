import time
import torch
import copy
import json
import os
from .base_trainer import BaseTrainer
from utils.flow_utils import evaluate_flow
from utils.misc_utils import AverageMeter
import utils.img_utils as img_utils

from external.perception_lib import viewer
from kittiloader import batch_scheduler
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import sys

class DefaultTrainer(BaseTrainer):
    def __init__(self, id, model, loss_func, _log, save_root, config, shared):
        super(DefaultTrainer, self).__init__(id, model, loss_func, _log, save_root, config, shared)
        self.cfg = config
        self.id = id

        # JSON
        if self.id == 0:
            self.foutput = {
                "name": self.cfg.data.exp_name,
                "rmse": [], "rmse_refined": [], "sil": [], "sil_refined": [], "rmse_unc": []
            }

        # Visualizer
        self.viz = None
        if self.cfg.var.viz and self.id == 0:
            self.viz = viewer.Visualizer("V")
            self.viz.start()

        # Create all variables here
        self.d_candi = img_utils.powerf(self.cfg.var.d_min, self.cfg.var.d_max, self.cfg.var.ndepth,
                                        self.cfg.var.qpower)
        self.d_candi_up = img_utils.powerf(self.cfg.var.d_min, self.cfg.var.d_max, self.cfg.var.ndepth * 2,
                                           self.cfg.var.qpower)

        # Create Data Loaders over here
        if self.cfg.mp.enabled:
            total_processes = self.cfg.mp.workers
            process_num = self.id
            batch_size = int(self.cfg.train.batch_size / total_processes)
        else:
            total_processes = 1
            process_num = 0
            batch_size = self.cfg.train.batch_size
        train_loader_params = {
            "pytorch_scaling": True,
            "velodyne_depth": True,
            "dataset_path": self.cfg.data.dataset_path,
            "split_path": self.cfg.data.dataset_split,
            "total_processes": total_processes,
            "process_num": process_num,
            "t_win_r": self.cfg.var.t_win,
            "img_size": self.cfg.var.img_size,
            "crop_w": self.cfg.var.crop_w,
            "d_candi": self.d_candi,
            "d_candi_up": self.d_candi_up,
            "dataset_name": "kitti",
            "hack_num": 0,
            "batch_size": batch_size,
            "n_epoch": 1,
            "qmax": 1,
            "mode": "train",
            "cfg": self.cfg
        }
        val_loader_params = copy.deepcopy(train_loader_params)
        val_loader_params["mode"] = "val"
        self._log.info(self.id, "=> Setting Batch Size {}.".format(batch_size))
        self.train_loader = batch_scheduler.BatchSchedulerMP(train_loader_params, self.cfg.var.mload)
        self.val_loader = batch_scheduler.BatchSchedulerMP(val_loader_params, self.cfg.var.mload)
        self.prev_output = None
        self.first_run = True

        # Viz
        self.model.module.set_viz(self.viz)

    def __del__(self):
        if self.viz is not None:
            self.viz.kill_received = True

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()
        key_meter_names = ['Loss', 'l_ph', 'l_sm', 'flow_mean']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        # Zero out shared
        self.shared[:] = 0

        # Model Train
        self.model.train()

        # Iterate Train Loader
        early_stop = False
        for items in self.train_loader.enumerate():
            if early_stop: break

            # ** Signal **
            if self.cfg.mp.enabled:
                signal = torch.tensor([1]).to(self.device)
                dist.all_reduce(signal)
                if signal.item() < self.cfg.mp.workers:
                    #self._log.info(self.id, "EXIT: " + str(signal.item()))
                    self.train_loader.stop()
                    early_stop = True
                    continue

            # Get data
            local_info, batch_length, batch_idx, frame_count, frame_length, iepoch = items
            if local_info['is_valid'].sum() == 0:
                loader_str = 'Corrupted Data! batch %d / %d, iter: %d, frame_count: %d / %d; Epoch: %d / %d' \
                % (batch_idx + 1, batch_length, self.i_iter, frame_count + 1, frame_length, self.i_epoch + 1, self.cfg.train.epoch_num)
                self._log.info(self.id, loader_str)
                continue

            # Reset Stage
            if frame_count == 0:
                self._log.info(self.id, "Reset Previous Output")
                self.prev_output = {"left": None, "right": None}

            # Create inputs
            local_info["d_candi"] = self.d_candi
            local_info["d_candi_up"] = self.d_candi_up
            if "stereo" in self.cfg["var"]:
                model_input_left, gt_input_left = batch_scheduler.generate_stereo_input(self.id, local_info, self.cfg, "left")
                model_input_right, gt_input_right = batch_scheduler.generate_stereo_input(self.id, local_info, self.cfg, "right")
            else:
                model_input_left, gt_input_left = batch_scheduler.generate_model_input(self.id, local_info, self.cfg, "left")
                model_input_right, gt_input_right = batch_scheduler.generate_model_input(self.id, local_info, self.cfg, "right")
            model_input_left["prev_output"] = self.prev_output["left"]
            model_input_right["prev_output"] = self.prev_output["right"]
            model_input_left["epoch"] = self.i_epoch; model_input_right["epoch"] = self.i_epoch

            # Model
            output_left, output_right = self.model([model_input_left, model_input_right])

            # Set Prev from last one
            output_left_intp = F.interpolate(output_left["output_refined"][-1].detach(), scale_factor=0.25, mode='nearest')
            output_right_intp = F.interpolate(output_right["output_refined"][-1].detach(), scale_factor=0.25, mode='nearest')
            self.prev_output = {"left": output_left_intp, "right": output_right_intp}

            # Loss Function
            loss = self.loss_func([output_left, output_right], [gt_input_left, gt_input_right])

            # Opt
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # String
            loader_str = 'Train batch %d / %d, iter: %d, frame_count: %d / %d; Epoch: %d / %d, loss = %.5f' \
            % (batch_idx + 1, batch_length, self.i_iter, frame_count + 1, frame_length, self.i_epoch + 1, self.cfg.train.epoch_num, loss)
            self._log.info(self.id, loader_str)
            self.i_iter += 1

        # ** Signal **
        if self.cfg.mp.enabled:
            #self._log.info(self.id, "AR: " + str(signal.item()))
            if signal.item() >= self.cfg.mp.workers:
                dist.all_reduce(torch.tensor([0]).to(self.device))
            #dist.barrier()

        self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):
        batch_time = AverageMeter()
        if self.cfg.mp.enabled:
            dist.barrier()

        # Zero out shared
        self.shared[:] = 0

        # Eval Mode
        self.model.eval()

        # Variables
        errors = []
        errors_refined = []
        errors_uncfield_rmse = []

        # Iterate Train Loader
        for items in self.val_loader.enumerate():

            # Get data
            local_info, batch_length, batch_idx, frame_count, frame_length, iepoch = items
            if local_info['is_valid'].sum() == 0:
                loader_str = 'Corrupted Data! Val batch %d / %d, frame_count: %d / %d' \
                % (batch_idx + 1, batch_length, frame_count + 1, frame_length)
                self._log.info(self.id, loader_str)
                continue

            # Reset Stage
            if frame_count == 0:
                self._log.info(self.id, "Reset Previous Output")
                self.prev_output = {"left": None, "right": None}

            # Create inputs
            local_info["d_candi"] = self.d_candi
            local_info["d_candi_up"] = self.d_candi_up
            if "stereo" in self.cfg["var"]:
                model_input_left, gt_input_left = batch_scheduler.generate_stereo_input(self.id, local_info, self.cfg, "left")
            else:
                model_input_left, gt_input_left = batch_scheduler.generate_model_input(self.id, local_info, self.cfg, "left")
            model_input_left["prev_output"] = self.prev_output["left"]
            model_input_left["epoch"] = self.i_epoch # Not sure if this will work during runtime/eval

            # Model
            start = time.time()
            output_left = self.model([model_input_left])[0]
            #output_right = self.model(model_input_right)
            print("Forward: " + str(time.time() - start))

            # Set Prev
            output_left_intp = F.interpolate(output_left["output_refined"][-1].detach(), scale_factor=0.25, mode='nearest')
            self.prev_output = {"left": output_left_intp, "right": None}

            # Visualization
            if self.cfg.var.viz:
                self.visualize(model_input_left, gt_input_left, output_left)

            # Eval
            for b in range(0, output_left["output"][-1].shape[0]):
                dpv_predicted = output_left["output"][-1][b, :, :, :].unsqueeze(0)
                dpv_refined_predicted = output_left["output_refined"][-1][b, :, :, :].unsqueeze(0)
                depth_predicted = img_utils.dpv_to_depthmap(dpv_predicted, self.d_candi, BV_log=True)
                depth_refined_predicted = img_utils.dpv_to_depthmap(dpv_refined_predicted, self.d_candi, BV_log=True)
                mask = gt_input_left["masks"][b, :, :, :]
                mask_refined = gt_input_left["masks_imgsizes"][b, :, :, :]
                depth_truth = gt_input_left["dmaps"][b, :, :].unsqueeze(0)
                depth_refined_truth = gt_input_left["dmap_imgsizes"][b, :, :].unsqueeze(0)
                dpv_refined_truth = gt_input_left["soft_labels_imgsize"][b].unsqueeze(0)
                d_candi = model_input_left["d_candi"]
                intr_refined = model_input_left["intrinsics_up"][b, :, :]

                # Unc Field
                unc_field_truth, unc_field_predicted, debugmap = img_utils.compute_unc_field(dpv_refined_predicted, dpv_refined_truth, d_candi, intr_refined, mask_refined, self.cfg)
                unc_field_rmse = img_utils.compute_unc_rmse(unc_field_truth, unc_field_predicted, d_candi)
                
                # Eval
                depth_truth_eval = depth_truth.clone()
                depth_truth_eval[depth_truth_eval >= self.d_candi[-1]] = self.d_candi[-1]
                depth_refined_truth_eval = depth_refined_truth.clone()
                depth_refined_truth_eval[depth_refined_truth_eval >= self.d_candi[-1]] = self.d_candi[-1]
                depth_predicted_eval = depth_predicted.clone()
                depth_predicted_eval = depth_predicted_eval * mask
                depth_refined_predicted_eval = depth_refined_predicted.clone()
                depth_refined_predicted_eval = depth_refined_predicted_eval * mask_refined
                errors.append(img_utils.depth_error(depth_predicted_eval.squeeze(0).cpu().numpy(), depth_truth_eval.squeeze(0).cpu().numpy()))
                errors_refined.append(img_utils.depth_error(depth_refined_predicted_eval.squeeze(0).cpu().numpy(), depth_refined_truth_eval.squeeze(0).cpu().numpy()))
                errors_uncfield_rmse.append(unc_field_rmse.item())

            # String
            loader_str = 'Val batch %d / %d, frame_count: %d / %d' \
            % (batch_idx + 1, batch_length, frame_count + 1, frame_length)
            self._log.info(self.id, loader_str)
            self.i_iter += 1

        # Evaluate Errors
        results = img_utils.eval_errors(errors)
        results_refined = img_utils.eval_errors(errors_refined)
        sil = results["scale invariant log"][0]
        sil_refined = results_refined["scale invariant log"][0]
        rmse = results["rmse"][0]
        rmse_refined = results_refined["rmse"][0]
        rmse_unc = np.mean(np.array(errors_uncfield_rmse))
        error_keys = ["rmse", "rmse_refined", "sil", "sil_refined", "rmse_unc"]
        error_list = [rmse, rmse_refined, sil, sil_refined, rmse_unc]

        # Copy to Shared
        for i, e in enumerate(error_list):
            self.shared[self.id, i] = e

        # Mean Error Compute Only First Process
        if self.cfg.mp.enabled:
            dist.barrier()
        error_list = torch.mean(self.shared, dim=0)
        if self.cfg.eval:
            print(error_list)

        # Save Model (Only First ID)
        self.save_model(rmse_refined, self.cfg.data.exp_name)

        # Log
        if self.id == 0:
            json_loc = str(self.save_root) + "/" + self.cfg.data.exp_name + '.json'

            # First Run
            if self.first_run:
                # Remove Results JSON if first epoch
                if self.i_epoch == 1:
                    if os.path.exists(json_loc):
                        os.remove(json_loc)
                # If not first epoch, then load past results
                else:
                    if os.path.isfile(json_loc):
                        with open(json_loc) as f:
                            self.foutput = json.load(f)

            # Save
            for value, name in zip(error_list, error_keys):
                self.foutput[name].append(value.item())
            with open(json_loc, 'w') as f:
                json.dump(self.foutput, f)

            # Tensorboard
            if self.summary_writer is not None:
                for value, name in zip(error_list, error_keys):
                    self.summary_writer.add_scalar(name, value, self.i_epoch)
                self.summary_writer.flush()

        # Set fr
        self.first_run = False

        return error_list, error_keys
        
    def visualize(self, model_input, gt_input, output):
        import cv2

        # Eval
        for b in range(0, output["output"][-1].shape[0]):
            dpv_predicted = output["output"][-1][b, :, :, :].unsqueeze(0)
            dpv_refined_predicted = output["output_refined"][-1][b, :, :, :].unsqueeze(0)
            d_candi = model_input["d_candi"]

            # Variance
            z = torch.exp(dpv_refined_predicted.squeeze(0))
            d_candi_expanded = torch.tensor(d_candi).unsqueeze(1).unsqueeze(1).cuda()
            mean = torch.sum(d_candi_expanded * z, dim=0)
            variance = torch.sum(((d_candi_expanded - mean) ** 2) * z, dim=0)
            # mask_var = (variance < 8.0).float()
            # dpv_refined_predicted = dpv_refined_predicted * mask_var.unsqueeze(0)

            # Convert
            depth_predicted = img_utils.dpv_to_depthmap(dpv_predicted, self.d_candi, BV_log=True)
            depth_refined_predicted = img_utils.dpv_to_depthmap(dpv_refined_predicted, self.d_candi, BV_log=True)
            mask = gt_input["masks"][b, :, :, :]
            mask_refined = gt_input["masks_imgsizes"][b, :, :, :]
            depth_truth = gt_input["dmaps"][b, :, :].unsqueeze(0)
            depth_refined_truth = gt_input["dmap_imgsizes"][b, :, :].unsqueeze(0)
            intr = model_input["intrinsics"][b, :, :]
            intr_refined = model_input["intrinsics_up"][b, :, :]
            img_refined = model_input["rgb"][b, -1, :, :, :].cpu()  # [1,3,256,384]
            img = F.interpolate(img_refined.unsqueeze(0), scale_factor=0.25, mode='bilinear').squeeze(0)
            dpv_refined_truth = gt_input["soft_labels_imgsize"][b].unsqueeze(0)

            # Eval
            tophalf_refined = torch.ones(depth_refined_predicted.shape).bool().to(depth_refined_predicted.device)
            tophalf_refined[:, 0:int(tophalf_refined.shape[1] / 3), :] = False
            depth_truth_eval = depth_truth.clone()
            depth_truth_eval[depth_truth_eval >= self.d_candi[-1]] = self.d_candi[-1]
            depth_refined_truth_eval = depth_refined_truth.clone()
            depth_refined_truth_eval[depth_refined_truth_eval >= self.d_candi[-1]] = self.d_candi[-1]
            depth_predicted_eval = depth_predicted.clone()
            depth_refined_predicted_eval = depth_refined_predicted.clone()
            depth_refined_predicted_eval = depth_refined_predicted_eval * tophalf_refined.float()
            img_color = img_utils.torchrgb_to_cv2(img_refined)
            img_color_low = img_utils.torchrgb_to_cv2(img)

            # UField
            _, unc_field_predicted, debugmap = img_utils.compute_unc_field(dpv_refined_predicted, dpv_refined_truth, d_candi, intr_refined, mask_refined, self.cfg)
            dmap_temp = F.interpolate(gt_input["dmaps"][b,:,:].unsqueeze(0).unsqueeze(0), scale_factor=4, mode='nearest').squeeze(0)
            true_dpv = img_utils.gen_dpv_withmask(dmap_temp, mask_refined.unsqueeze(0)*0+1, d_candi, 0.3)
            unc_field_truth, _ = img_utils.gen_ufield(true_dpv, d_candi, intr_refined.squeeze(0), BV_log=False, cfg=self.cfg)
            field_visual = np.zeros((unc_field_truth.shape[1], unc_field_truth.shape[2], 3))
            field_visual[:,:,0] = unc_field_predicted[0,:,:].cpu().numpy()*3
            field_visual[:,:,1] = unc_field_predicted[0,:,:].cpu().numpy()*3
            field_visual[:,:,2] = unc_field_truth[0,:,:].cpu().numpy()*3
            field_visual = cv2.resize(field_visual, None, fx=1, fy=2, interpolation = cv2.INTER_CUBIC)
            img_color[:, :, 0] += debugmap.squeeze(0).cpu().numpy()
            img_color = np.clip(img_color, 0, 1)

            # Expand it for Display
            field_visual = cv2.flip(field_visual, 0 )
            field_visual[0:5,:,:] = 0
            field_visual_expanded = np.zeros(img_color.shape)
            field_visual_expanded[field_visual_expanded.shape[0]-field_visual.shape[0]:field_visual_expanded.shape[0], :, :] = field_visual
            field_visual_expanded = np.clip(field_visual_expanded, 0, 1)
            field_visual_expanded[np.isnan(field_visual_expanded)] = 0

            # Display Image/Depth
            img_depth = cv2.cvtColor(depth_refined_predicted_eval[0, :, :].cpu().numpy() / 100., cv2.COLOR_GRAY2BGR)
            img_depth_low = cv2.cvtColor(depth_predicted_eval[0, :, :].cpu().numpy() / 100., cv2.COLOR_GRAY2BGR)
            truth_depth = cv2.cvtColor(depth_refined_truth_eval[0, :, :].cpu().numpy() / 100., cv2.COLOR_GRAY2BGR)
            truth_depth_low = cv2.cvtColor(depth_truth_eval[0, :, :].cpu().numpy() / 100., cv2.COLOR_GRAY2BGR)
            combined_low = np.hstack([img_color_low, img_depth_low, truth_depth_low])
            combined = np.hstack([img_color, img_depth, truth_depth])
            #combined = np.hstack([img_color, truth_depth])
            #combined_low = np.hstack([img_color_low, truth_depth_low])
            #cv2.imshow("combined", combined)
            #cv2.imshow("combined_low", combined_low)

            # Field
            # if self.i_iter == 0:
            #     cv2.imshow("field_visual", field_visual)
            # Field Visual
            field_visual_expanded = (field_visual_expanded - np.min(field_visual_expanded))/(np.max(field_visual_expanded) - np.min(field_visual_expanded))
            field_visual_expanded = (field_visual_expanded * 255).astype(np.uint8)
            #cv2.imshow("field_visual_expanded", field_visual_expanded)
            # RGB
            img_color = (img_color - np.min(img_color))/(np.max(img_color) - np.min(img_color))
            img_color = (img_color * 255).astype(np.uint8)
            #cv2.imshow("imR", img_color)
            # Depth Heatmap
            dmap = depth_refined_predicted_eval[0, :, :].cpu().numpy()
            dmap = ((dmap/np.max(dmap))*255).astype(np.uint8)
            imC = cv2.applyColorMap(dmap, cv2.COLORMAP_JET)
            #imC = (imC/255).astype(np.float32)
            #cv2.imshow("imC", imC)
            # Depth errormap
            error_map = depth_refined_truth_eval[0,:,:] - depth_refined_predicted_eval[0, :, :]*mask_refined
            error_map = F.interpolate(error_map.unsqueeze(0), scale_factor=0.25, mode='bilinear')
            error_map = F.interpolate(error_map, scale_factor=4, mode='bicubic').squeeze(0)
            error_map = error_map.squeeze(0).cpu().numpy()
            error_map = np.clip(((error_map/1)*255),0,255).astype(np.uint8)
            error_map = cv2.cvtColor(error_map, cv2.COLOR_GRAY2BGR)
            #error_map = (error_map/255).astype(np.float32)
            #cv2.imshow("error_map", error_map)
            # Variance map
            variance_numpy = variance.cpu().numpy()
            variance_maxval = np.max(variance_numpy)
            variance_map = np.clip(((variance_numpy/50)*255),0,255).astype(np.uint8)
            imV = cv2.applyColorMap(variance_map, cv2.COLORMAP_JET)
            #imV = (imV/255).astype(np.float32)
            #cv2.imshow("imV", imV)
            # Combined
            combined = np.hstack([img_color, imC, imV, field_visual_expanded])
            cv2.imshow("combined", combined)

            # Shift
            depth_refined_truth_eval = img_utils.cull_depth(depth_refined_truth_eval, intr_refined)
            #depth_refined_predicted_eval = img_utils.cull_depth(depth_refined_predicted_eval, intr_refined)
            depth_truth_eval = img_utils.cull_depth(depth_truth_eval, intr, 3)

            # Cloud
            cloud_truth = img_utils.tocloud(depth_truth_eval, img_utils.demean(img), intr, None, [255,255,0])
            cloud_predicted = img_utils.tocloud(depth_predicted_eval, img_utils.demean(img), intr)
            cloud_refined_truth = img_utils.tocloud(depth_refined_truth_eval, img_utils.demean(img_refined), intr_refined, None, [255,255,0])
            cloud_refined_predicted = img_utils.tocloud(depth_refined_predicted_eval, img_utils.demean(img_refined), intr_refined)
            self.viz.addCloud(cloud_refined_predicted, 3)
            self.viz.addCloud(cloud_truth, 5)

            # Swap
            self.viz.swapBuffer()
            key = -1
            if self.i_iter == 0:
                cv2.waitKey(0)
            else:
                key = cv2.waitKey(5)

            # Render Viz
            rendered_img = self.viz.getRenderedImage()

            # Write Video
            write_video = self.cfg.write_video
            if len(write_video):
                if self.i_iter == 0:
                    self.images_out = cv2.VideoWriter(write_video + "/" + "images" + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (combined.shape[1],combined.shape[0]))
                    self.rendered_out = cv2.VideoWriter(write_video + "/" + "rendered" + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (rendered_img.shape[1],rendered_img.shape[0]))
                self.images_out.write(combined)
                self.rendered_out.write(rendered_img)
                if key == 113:
                    self.images_out.release()
                    self.rendered_out.release()
                    sys.exit(0)

        pass