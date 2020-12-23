import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import utils.img_utils as img_utils
import cv2
import matplotlib.pyplot as plt
import time
from lc import light_curtain
import copy
import sys
import rospy
from external.perception_lib import viewer

sys.path.append("external/lcsim/python")
from sim import LCDevice
from planner import PlannerRT
import pylc_lib as pylc

import cv2
import torch
import numpy as np
import pickle
import os
import rospy
import rospkg
import sys
import json
import tf
import copy
import multiprocessing
import time
import shutil

import threading
from collections import deque
import functools
import message_filters
import os
import rospy
from message_filters import ApproximateTimeSynchronizer
import sensor_msgs.msg
import sensor_msgs.srv
from tf.transformations import quaternion_matrix
import image_geometry
import cv_bridge
from cv_bridge import CvBridge
devel_folder = rospkg.RosPack().get_path('params_lib').split("/")
devel_folder = '/'.join(devel_folder[:len(devel_folder)-2]) + "/devel/lib/"
sys.path.append(devel_folder)
import params_lib_python

from easydict import EasyDict
from models.get_model import get_model
from utils.torch_utils import bias_parameters, weight_parameters, \
    load_checkpoint, save_checkpoint, AdamW
import warping.view as View
import torchvision.transforms as transforms

from sensor_bridge.msg import TensorMsg
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import zlib

from ros_all import *

class ConsumerThread(threading.Thread):
    def __init__(self, queue, function):
        threading.Thread.__init__(self)
        self.queue = queue
        self.function = function

    def run(self):
        while True:
            # wait for an image (could happen at the very beginning when the queue is still empty)
            while len(self.queue) == 0:
                time.sleep(0.1)
            self.function(self.queue[0])

class RosNet():
    def __init__(self):
        self.transforms = None
        self.index = 0
        self.prev_index = -1
        self.bridge = CvBridge()
        self.prev_seq = None
        self.just_started = True
        self.mode = rospy.get_param('~mode', 'stereo')
        self.plan = rospy.get_param('~plan', 0)
        self.return_mode = rospy.get_param('~return_mode', 0)
        print(self.mode, self.plan)
        params_file = 'real_sensor.json'

        # Planner
        self.planner = None
        if self.plan:
            self.planner = Planner(mode="real", params_file=params_file)

        #  Params
        with open(params_file) as f:
            self.param = json.load(f)
        self.param["d_candi"] = img_utils.powerf(self.param["s_range"], self.param["e_range"], 64, 1.)

        # Gen Model Datum
        intrinsics = torch.tensor(self.param["intr_rgb"]).unsqueeze(0)/4; intrinsics[0,2,2] = 1.
        intrinsics_up = torch.tensor(self.param["intr_rgb"]).unsqueeze(0)
        s_width = self.param["size_rgb"][0]/4
        s_height = self.param["size_rgb"][1]/4
        focal_length = np.mean([intrinsics_up[0,0,0], intrinsics_up[0,1,1]])
        h_fov = math.degrees(math.atan(intrinsics_up[0,0, 2] / intrinsics_up[0,0, 0]) * 2)
        v_fov = math.degrees(math.atan(intrinsics_up[0,1, 2] / intrinsics_up[0,1, 1]) * 2)
        pixel_to_ray_array = View.normalised_pixel_to_ray_array(\
                width= int(s_width), height= int(s_height), hfov = h_fov, vfov = v_fov,
                normalize_z = True)
        pixel_to_ray_array_2dM = np.reshape(np.transpose( pixel_to_ray_array, axes= [2,0,1] ), [3, -1])
        pixel_to_ray_array_2dM = torch.from_numpy(pixel_to_ray_array_2dM.astype(np.float32)).unsqueeze(0)
        left_2_right = torch.tensor(self.param["left_2_right"])
        if self.mode == "stereo" or self.mode == "stereo_lc":
            src_cam_poses = torch.cat([left_2_right.unsqueeze(0), torch.eye(4).unsqueeze(0)]).unsqueeze(0)
        elif self.mode == "mono" or self.mode == "mono_lc":
            src_cam_poses = torch.cat([torch.eye(4).unsqueeze(0), torch.eye(4).unsqueeze(0)]).unsqueeze(0)
        self.model_datum = dict()
        self.model_datum["intrinsics"] = intrinsics.cuda()
        self.model_datum["intrinsics_up"] = intrinsics_up.cuda()
        self.model_datum["unit_ray"] = pixel_to_ray_array_2dM.cuda()
        self.model_datum["src_cam_poses"] = src_cam_poses.cuda()
        self.model_datum["d_candi"] = self.param["d_candi"]
        self.model_datum["d_candi_up"] = self.param["d_candi"]
        self.model_datum["rgb"] = None
        self.model_datum["prev_output"] = None
        self.model_datum["prev_lc"] = None
        self.rgb_pinned = torch.zeros((1,2,3,self.param["size_rgb"][1], self.param["size_rgb"][0])).float().pin_memory()
        self.dpv_pinned = torch.zeros((1,64,int(self.param["size_rgb"][1]), int(self.param["size_rgb"][0]))).float().pin_memory()
        self.pred_depth_pinned = torch.zeros((int(self.param["size_rgb"][1]), int(self.param["size_rgb"][0]))).float().pin_memory()
        self.true_depth_pinned = torch.zeros((int(self.param["size_rgb"][1]), int(self.param["size_rgb"][0]))).float().pin_memory()
        self.unc_pinned = torch.zeros(1,64, int(self.param["size_rgb"][0])).float().pin_memory()
        __imagenet_stats = {'mean': [0.485, 0.456, 0.406],\
                            'std': [0.229, 0.224, 0.225]}
        self.transformer = transforms.Normalize(**__imagenet_stats)

        # Load Model
        if self.mode == "stereo":
            model_name = 'default_stereo_ilim'
        elif self.mode == "mono":
            model_name = 'default_ilim'
        elif self.mode == "mono_lc":
            model_name = 'default_exp7_lc_ilim'
        elif self.mode == 'stereo_lc':
            model_name = 'default_stereo_exp7_lc_ilim'
        cfg_path = 'configs/' + model_name + '.json'
        model_path = ''
        with open(cfg_path) as f:
            self.cfg = EasyDict(json.load(f))
        self.model = get_model(self.cfg, 0)
        epoch, weights = load_checkpoint('outputs/checkpoints/' + model_name + '/' + model_name + '_model_best.pth.tar')
        from collections import OrderedDict
        new_weights = OrderedDict()
        model_keys = list(self.model.state_dict().keys())
        weight_keys = list(weights.keys())
        for a, b in zip(model_keys, weight_keys):
            new_weights[a] = weights[b]
        weights = new_weights
        self.model.load_state_dict(weights)
        self.model = self.model.cuda()
        self.model.eval()
        print("Model Loaded")

        # ROS
        self.q_msg = deque([], 1)
        lth = ConsumerThread(self.q_msg, self.handle_msg)
        lth.setDaemon(True)
        lth.start()
        self.queue_size = 3
        self.sync = functools.partial(ApproximateTimeSynchronizer, slop=0.01)
        self.left_camsub = message_filters.Subscriber('/left_camera_resized/image_color_rect', sensor_msgs.msg.Image)
        self.right_camsub = message_filters.Subscriber('right_camera_resized/image_color_rect', sensor_msgs.msg.Image)
        self.depth_sub = message_filters.Subscriber('/left_camera_resized/depth', sensor_msgs.msg.Image) # , queue_size=self.queue_size, buff_size=2**24
        self.ts = self.sync([self.left_camsub, self.right_camsub, self.depth_sub], self.queue_size)
        self.ts.registerCallback(self.callback)
        self.prev_left_cammsg = None
        self.depth_pub = rospy.Publisher('ros_net/depth', sensor_msgs.msg.Image, queue_size=self.queue_size)
        self.depth_color_pub = rospy.Publisher('ros_net/depth_color', sensor_msgs.msg.Image, queue_size=self.queue_size)
        self.depth_lc_pub = rospy.Publisher('ros_net/depth_lc', sensor_msgs.msg.Image, queue_size=self.queue_size)
        self.dpv_pub = rospy.Publisher('ros_net/dpv_pub', TensorMsg, queue_size=self.queue_size)
        self.unc_pub = rospy.Publisher('ros_net/unc_pub', TensorMsg, queue_size=self.queue_size)
        self.debug_pub = rospy.Publisher('ros_net/debug', sensor_msgs.msg.Image, queue_size=self.queue_size)
        self.debug2_pub = rospy.Publisher('ros_net/debug2', sensor_msgs.msg.Image, queue_size=self.queue_size)
        self.sensed_pub = rospy.Publisher('ros_net/sensed_pub', TensorMsg, queue_size=self.queue_size)

    def gen_rgb_tensor(self, inp1, inp2):
        inp1 = torch.tensor(inp1).permute(2,0,1)
        inp1 = inp1[[2,1,0], :, :]
        inp1 = self.transformer(inp1)
        inp2 = torch.tensor(inp2).permute(2,0,1)
        inp2 = inp2[[2,1,0], :, :]
        inp2 = self.transformer(inp2)
        self.rgb_pinned[:] = torch.cat([inp1.unsqueeze(0), inp2.unsqueeze(0)]).unsqueeze(0)
        #self.rgb_pinned[0, 0, :, :, :] = inp1
        #self.rgb_pinned[0, 1, :, :, :] = inp2
        self.model_datum["rgb"] = self.rgb_pinned.cuda(non_blocking=True)
        
    def destroy(self):
        self.left_camsub.unregister()
        self.right_camsub.unregister()
        self.depth_sub.unregister()
        del self.ts

    def get_transform(self, from_frame, to_frame):
        listener = tf.TransformListener()
        listener.waitForTransform(to_frame, from_frame, rospy.Time(), rospy.Duration(4.0))
        (trans,rot) = listener.lookupTransform(to_frame, from_frame, rospy.Time(0))
        matrix = quaternion_matrix(rot)
        matrix[0:3, 3] = trans
        return matrix.astype(np.float32)

    def convert_imgmsg(self, cammsg, caminfomsg):
        cvImg = self.bridge.imgmsg_to_cv2(cammsg, desired_encoding='passthrough')
        cam_model = image_geometry.PinholeCameraModel()
        cam_model.fromCameraInfo(caminfomsg)
        cam_model.rectifyImage(cvImg, 0)
        cvImg = cv2.remap(cvImg, cam_model.mapx, cam_model.mapy, 1)
        return cvImg, cam_model

    def callback(self, left_cammsg, right_cammsg, depth_msg):
        # Append msg
        self.q_msg.append((left_cammsg, right_cammsg, self.prev_left_cammsg, depth_msg))

        # Store prev here
        self.prev_left_cammsg = left_cammsg

        # Index
        self.index += 1

    @torch.no_grad()
    def handle_msg(self, msg):
        if self.prev_index == self.index:
            time.sleep(0.00001)
            return
        self.prev_index = self.index
        left_cammsg, right_cammsg, prev_left_cammsg, depth_msg = msg
        print("enter")

        # Convert
        if self.mode == "mono" or self.mode == "mono_lc":
            if prev_left_cammsg == None:
                return
            prev_left_img = self.bridge.imgmsg_to_cv2(prev_left_cammsg, desired_encoding='passthrough').astype(np.float32)/255.
        left_img = self.bridge.imgmsg_to_cv2(left_cammsg, desired_encoding='passthrough').astype(np.float32)/255.
        right_img = self.bridge.imgmsg_to_cv2(right_cammsg, desired_encoding='passthrough').astype(np.float32)/255.
        depth_img = torch.tensor(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)/1000.)
        self.true_depth_pinned[:] = depth_img[:]
        true_depth_r_tensor = self.true_depth_pinned.cuda(non_blocking=True).unsqueeze(0)

        # Gen Tensor
        if self.mode == "stereo" or self.mode == "stereo_lc":
            self.gen_rgb_tensor(right_img, left_img)
        elif self.mode == "mono" or self.mode == "mono_lc":
            self.gen_rgb_tensor(prev_left_img, left_img)

        # Call Model
        output = self.model([self.model_datum])[0]
        dpv_refined_predicted = output["output_refined"][-1]

        # Save Prev
        self.model_datum["prev_output"] = F.interpolate(dpv_refined_predicted, scale_factor=0.25, mode='nearest')

        # Generate Depth
        depth_refined_predicted = img_utils.dpv_to_depthmap(dpv_refined_predicted, self.model_datum["d_candi"], BV_log=True)

        # UField Display
        #self.param["unc_shift"] = 1.3
        unc_field_predicted, debugmap = img_utils.gen_ufield(dpv_refined_predicted, self.model_datum["d_candi"], self.model_datum["intrinsics_up"].squeeze(0), BV_log=True, cfgx=self.param)

        # RGB Draw Debug
        rgb_debug = left_img.copy()
        rgb_debug[:,:,0] += debugmap[0,:,:].cpu().numpy()

        # Planner (Slow)
        depth_lc = None
        sensed_set = None
        if self.planner is not None:
            dpv_upsampled = img_utils.upsample_dpv(dpv_refined_predicted.clone(), N=self.planner.real_lc.expand_A, BV_log=True)
            self.planner.final = dpv_upsampled
            field_visual, depth_lc, sensed_set = self.planner.run(None, true_depth_r_tensor, return_mode=self.return_mode)
            dpv_lc_fused = img_utils.upsample_dpv(self.planner.final.clone(), N=64, BV_log=True)
            output_lc = F.interpolate(dpv_lc_fused.detach(), scale_factor=0.25, mode='nearest')
            self.model_datum["prev_lc"] = output_lc
            #pass
            #cv2.imshow("WIN", field_visual)
            #cv2.waitKey(1)

        # CPU
        #dpv_refined_predicted_cpu = dpv_refined_predicted.cpu().numpy()
        self.dpv_pinned[:] = dpv_refined_predicted[:]
        self.pred_depth_pinned[:] = depth_refined_predicted[:]
        self.unc_pinned[:] = unc_field_predicted[:]

        # Publish
        if self.sensed_pub.get_num_connections() and sensed_set is not None:
            ros_sensed = TensorMsg()
            ros_sensed.data = sensed_set.tostring()
            ros_sensed.shape = sensed_set.shape
            ros_sensed.header = left_cammsg.header
            self.sensed_pub.publish(ros_sensed)

        # Publish
        if self.dpv_pub.get_num_connections() or self.unc_pub.get_num_connections():
            ros_dpv = TensorMsg()
            ros_dpv.data = self.dpv_pinned.numpy().tostring()
            ros_dpv.shape = self.dpv_pinned.shape
            ros_dpv.header = left_cammsg.header
            ros_unc = TensorMsg()
            ros_unc.data = self.unc_pinned.numpy().tostring()
            ros_unc.shape = self.unc_pinned.shape
            ros_unc.header = left_cammsg.header
            self.dpv_pub.publish(ros_dpv)
            self.unc_pub.publish(ros_unc)

        if self.depth_pub.get_num_connections():
            ros_depth = ((self.pred_depth_pinned[:].squeeze(0).numpy()) * 1000).astype(np.uint16)
            ros_depth = self.bridge.cv2_to_imgmsg(ros_depth)
            ros_depth.header = left_cammsg.header
            self.depth_pub.publish(ros_depth)

        if self.depth_color_pub.get_num_connections():
            depthmap = self.pred_depth_pinned[:].squeeze(0).numpy()
            depthmap = ((depthmap/np.max(depthmap))*255).astype(np.uint8)
            depthmap = cv2.applyColorMap(depthmap, cv2.COLORMAP_JET)
            ros_depth = self.bridge.cv2_to_imgmsg(depthmap, encoding="rgb8")
            self.depth_color_pub.publish(ros_depth)

        if depth_lc is not None:
            if self.depth_lc_pub.get_num_connections():
                ros_depth = (depth_lc * 1000).astype(np.uint16)
                ros_depth = self.bridge.cv2_to_imgmsg(ros_depth)
                ros_depth.header = left_cammsg.header
                self.depth_lc_pub.publish(ros_depth)

        if self.debug_pub.get_num_connections():
            field_visual = self.unc_pinned.squeeze(0).numpy()*3
            debug = self.bridge.cv2_to_imgmsg((field_visual*255).astype(np.uint8), encoding="8UC1")
            debug.header = left_cammsg.header
            self.debug_pub.publish(debug)

        if self.debug2_pub.get_num_connections():
            debug = self.bridge.cv2_to_imgmsg((rgb_debug*255).astype(np.uint8), encoding="rgb8")
            debug.header = left_cammsg.header
            self.debug2_pub.publish(debug)

rospy.init_node('ros_net', anonymous=False)
rosnet = RosNet()
rospy.spin()