import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import warping.homography as warp_homo
import utils.img_utils as img_utils
from functools import partial

class resconv_basic(nn.Module):
    """
    Base residual convolution class
    """
    def __init__(self, in_planes, out_planes, stride, dropout=None):
        super(resconv_basic, self).__init__()
        self.out_planes = out_planes
        self.stride = stride
        self.conv1 = conv(in_planes, out_planes, 3, stride)
        self.conv2 = conv(out_planes, out_planes, 3, 1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)
        self.normalize = torch.nn.GroupNorm(16, out_planes)

        if dropout:
            self.conv3 = nn.Sequential(self.conv3, nn.Dropout2d(dropout))

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        shortcut = self.conv3(x)
        return F.elu(self.normalize(x_out + shortcut), inplace=True)

def resblock_basic(in_planes, out_planes, num_blocks, stride, dropout=None):
    """
    Base residual block class
    """
    layers = []
    layers.append(resconv_basic(in_planes, out_planes, stride, dropout=dropout))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(out_planes, out_planes, 1, dropout=dropout))
    return nn.Sequential(*layers)

class conv(nn.Module):
    """
    Base convolution 2D class
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                   stride=stride)
        self.normalize = torch.nn.GroupNorm(16, out_planes)
        p = kernel_size // 2
        self.p2d = (p, p, p, p)

    def forward(self, x):
        x = self.conv_base(F.pad(x, self.p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)

def packing(x, r=2):
    """Takes a BCHW tensor and returns a B(rC)(H/r)(W/r) tensor,
    by concatenating neighbor spatial pixels as extra channels.
    It is the inverse of nn.PixelShuffle (if you apply both sequentially you should get the same tensor)
    Example r=2: A RGB image (C=3) becomes RRRRGGGGBBBB (C=12) and is downsampled to half its size
    """
    b, c, h, w = x.shape
    out_channel = c * (r ** 2)
    out_h, out_w = h // r, w // r
    x = x.contiguous().view(b, c, out_h, r, out_w, r)
    return x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)

class PackLayerConv2d(nn.Module):
    """Packing layer with 2d convolutions.
    Takes a BCHW tensor, packs it into B4CH/2W/2 and then convolves it to
    produce BCH/2W/2.
    """
    def __init__(self, in_channels, kernel_size, r=2):
        super().__init__()
        self.conv = conv(in_channels * (r ** 2), in_channels, kernel_size, 1)
        self.pack = partial(packing, r=r)

    def forward(self, x):
        x = self.pack(x)
        x = self.conv(x)
        return x

class UnpackLayerConv2d(nn.Module):
    """Unpacking layer with 2d convolutions.
    Takes a BCHW tensor, convolves it to produce B4CHW and then unpacks it to
    produce BC2H2W.
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2):
        super().__init__()
        self.conv = conv(in_channels, out_channels * (r ** 2), kernel_size, 1)
        self.unpack = nn.PixelShuffle(r)

    def forward(self, x):
        x = self.conv(x)
        x = self.unpack(x)
        return x

class PackLayerConv3d(nn.Module):
    """Packing layer with 3d convolutions.
    Takes a BCHW tensor, packs it into B4CH/2W/2 and then convolves it to
    produce BCH/2W/2.
    """
    def __init__(self, in_channels, kernel_size, r=2, d=8):
        super().__init__()
        self.conv = conv(in_channels * (r ** 2) * d, in_channels, kernel_size, 1)
        self.pack = partial(packing, r=r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        x = self.pack(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.conv(x)
        return x

class UnpackLayerConv3d(nn.Module):
    """Unpacking layer with 3d convolutions.
    Takes a BCHW tensor, convolves it to produce B4CHW and then unpacks it to
    produce BC2H2W.
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2, d=8):
        super().__init__()
        self.conv = conv(in_channels, out_channels * (r ** 2) // d, kernel_size, 1)
        self.unpack = nn.PixelShuffle(r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.unpack(x)
        return x

class BaseEncoder(nn.Module):
    def __init__(self, cfg):
        '''
        inputs:
        multi_scale - if output multi-sclae features:
        [1/4 scale of input image, 1/2 scale of input image]
        '''

        super(BaseEncoder, self).__init__()
        self.cfg = cfg
        feature_dim = self.cfg.var.feature_dim
        D = self.cfg.var.ndepth

        # Hyper-parameters
        dropout = 0.5
        ni = int(feature_dim/2)
        no = feature_dim
        n1, n2, n3, n4, n5 = ni, no, no, no, no
        num_blocks = [3, 3, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]

        n1o, n1i = n1, n1 + ni + no
        n2o, n2i = n2, n2 + n1 + no
        n3o, n3i = n3, n3 + n2 + no
        n4o, n4i = n4, n4 + n3
        n5o, n5i = n5, n5 + n4

        self.pre_calc = conv(3, ni, 5, 1)

        # Encoder
        self.pack1 = PackLayerConv3d(n1, pack_kernel[0])
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1])
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2])
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3])
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4])
        self.conv1 = conv(ni, n1, 7, 1)
        self.conv2 = resblock_basic(n1, n2, num_blocks[0], 1, dropout=dropout)
        self.conv3 = resblock_basic(n2, n3, num_blocks[1], 1, dropout=dropout)
        self.conv4 = resblock_basic(n3, n4, num_blocks[2], 1, dropout=dropout)
        self.conv5 = resblock_basic(n4, n5, num_blocks[3], 1, dropout=dropout)
        toc = n2 + n3 + n4 + n5
        self.compress = nn.Sequential(nn.Conv2d(toc, int(toc/2), 3, stride=1, padding=1, bias=False),
                                      nn.GroupNorm(16, int(toc/2)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(int(toc/2), D, kernel_size=1, padding=0, stride=1, bias=False))
        #self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Conv3d)):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()

    def forward(self, x):
        xf = self.pre_calc(x)
        x1 = self.conv1(xf)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        ###
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)
        ###
        x3p_up = F.upsample(x3p, scale_factor=2, mode='bilinear', align_corners=True)
        x4p_up = F.upsample(x4p, scale_factor=4, mode='bilinear', align_corners=True)
        x5p_up = F.upsample(x5p, scale_factor=8, mode='bilinear', align_corners=True)
        output_feature = torch.cat((x2p, x3p_up, x4p_up, x5p_up), 1)
        compressed = self.compress(output_feature)

        # print("xf: " + str(xf.shape))
        # print("x1: " + str(x1.shape))
        # print("x1p: " + str(x1p.shape))
        # print("x2: " + str(x2.shape))
        # print("x2p: " + str(x2p.shape))
        # print("x3: " + str(x3.shape))
        # print("x3p: " + str(x3p.shape))
        # print("x4: " + str(x4.shape))
        # print("x4p: " + str(x4p.shape))
        # print("x5: " + str(x5.shape))
        # print("x5p: " + str(x5p.shape))
        # print("x3p_up: " + str(x3p_up.shape))
        # print("x4p_up: " + str(x4p_up.shape))
        # print("x5p_up: " + str(x5p_up.shape))

        return [x2p, x1p, xf], compressed

class BaseDecoder(nn.Module):
    def __init__(self, cfg):
        '''
        inputs:
        multi_scale - if output multi-sclae features:
        [1/4 scale of input image, 1/2 scale of input image]
        '''

        super(BaseDecoder, self).__init__()
        self.cfg = cfg
        feature_dim = self.cfg.var.feature_dim
        D = self.cfg.var.ndepth

        # Hyper-parameters
        dropout = 0.5
        ni = int(feature_dim/2)
        no = feature_dim
        n1, n2, n3, n4, n5 = ni, no, no, no, no
        num_blocks = [3, 3, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]

        n1o, n1i = n1, n1 + ni + no
        n2o, n2i = n2, n2 + n1 + no
        n3o, n3i = n3, n3 + n2 + no
        n4o, n4i = n4, n4 + n3
        n5o, n5i = n5, n5 + n4

        # Decoder
        self.unpack3 = UnpackLayerConv3d(64, 64, unpack_kernel[2])
        self.unpack2 = UnpackLayerConv3d(64, 64, unpack_kernel[3])

        self.iconv3 = conv(128, 64, iconv_kernel[2], 1)
        self.iconv2 = conv(96, 64, iconv_kernel[3], 1)
        self.iconv1 = conv(96, 64, iconv_kernel[4], 1)

        #self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Conv3d)):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()

    def forward(self, x, features):
        x2p, x1p, xf = features
        skip1 = xf
        skip2 = x1p
        skip3 = x2p

        concat3 = torch.cat([x, skip3], dim=1)
        iconv3 = self.iconv3(concat3)
        unpack3 = self.unpack3(iconv3)

        concat2 = torch.cat([unpack3, skip2], dim=1)
        iconv2 = self.iconv2(concat2)
        unpack2 = self.unpack2(iconv2)

        concat1 = torch.cat([unpack2, skip1], dim=1)
        iconv1 = self.iconv1(concat1)

        dpv_refined = F.log_softmax(iconv1, dim=1)

        return dpv_refined

class PacknetModel(nn.Module):
    def __init__(self, cfg, id):
        super(PacknetModel, self).__init__()
        self.cfg = cfg
        self.id = id
        self.sigma_soft_max = self.cfg.var.sigma_soft_max
        self.feature_dim = self.cfg.var.feature_dim
        self.nmode = self.cfg.var.nmode
        D = self.cfg.var.ndepth

        self.base_encoder = BaseEncoder(self.cfg)
        self.base_decoder = BaseDecoder(self.cfg)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[1]
            factor = (n+1) // 2
            if n %2 ==1:
                center = factor - 1
            else:
                center = factor -.5
            og = np.ogrid[:n, :n]
            weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            m.weight.data.copy_(torch.from_numpy(weights_np))

    def init_weights(self):
        self.apply(self.weight_init)

    def forward_encoder(self, model_input):
        bsize = model_input["rgb"].shape[0]
        d_candi = model_input["d_candi"]

        # Feature Extraction
        rgb = model_input["rgb"]
        rgb_reshaped = rgb.view(rgb.shape[0]*rgb.shape[1], rgb.shape[2], rgb.shape[3], rgb.shape[4])
        feats_raw, feat_imgs = self.base_encoder(rgb_reshaped)

        # Feat Imgs
        dw_rate = int(rgb_reshaped.shape[3] / feat_imgs.shape[3])
        img_features = F.avg_pool2d(rgb_reshaped, dw_rate)
        feat_imgs_all = torch.cat( (feat_imgs, img_features), dim=1 )
        feat_imgs_all = feat_imgs_all.view(rgb.shape[0], rgb.shape[1], feat_imgs_all.shape[1], feat_imgs_all.shape[2], feat_imgs_all.shape[3])

        # Reshape Features (Based on twin)
        twin = feat_imgs_all.shape[1]
        feature_set = []
        for tw in range(0, twin): feature_set.append([])
        for feat_raw in feats_raw:
            feat_raw_reshaped = feat_raw.view(rgb.shape[0], rgb.shape[1], feat_raw.shape[1], feat_raw.shape[2], feat_raw.shape[3])
            for tw in range(0, twin):
                feature_set[tw].append(feat_raw_reshaped[:,tw,:,:,:])

        # Warp Cost Volume for each video batch
        cost_volumes = []
        for i in range(0, bsize):
            Rs_src = model_input["src_cam_poses"][i,:-1, :3,:3]
            ts_src = model_input["src_cam_poses"][i,:-1, :3,3]

            feat_img_ref = feat_imgs_all[i,-1,:,:,:].unsqueeze(0)
            feat_imgs_src = feat_imgs_all[i,:-1,:,:,:].unsqueeze(0)

            cam_intrinsics = {"intrinsic_M_cuda": model_input["intrinsics"][i,:,:],
                              "intrinsic_M": model_input["intrinsics"][i,:,:].cpu().numpy(),
                              "unit_ray_array_2D": model_input["unit_ray"][i,:,:]}

            costV = warp_homo.est_swp_volume_v4( \
                    feat_img_ref,
                    feat_imgs_src,
                    d_candi, Rs_src, ts_src,
                    cam_intrinsics,
                    self.sigma_soft_max,
                    feat_dist = 'L2')

            cost_volumes.append(costV)
        cost_volumes = torch.cat(cost_volumes, dim=0) # [4 128 64 96]

        # Some Conv Here?

        # Log it
        BV = F.log_softmax(cost_volumes, dim=1)

        return BV, feature_set

    def forward(self, input):

        BV_cur, feature_set = self.forward_encoder(input)

        BV_cur_refined = self.base_decoder(torch.exp(BV_cur), feature_set[-1])

        return {"output": [BV_cur], "output_refined": [BV_cur_refined], "flow": None, "flow_refined": None}

        pass