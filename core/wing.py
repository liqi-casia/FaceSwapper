"""
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""


from functools import partial
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

def normalize(x, eps=1e-6):
    """Apply min-max normalization."""
    x = x.contiguous()
    N, C, H, W = x.size()
    x_ = x.view(N*C, -1)
    max_val = torch.max(x_, dim=1, keepdim=True)[0]
    min_val = torch.min(x_, dim=1, keepdim=True)[0]
    x_ = (x_ - min_val) / (max_val - min_val + eps)
    out = x_.view(N, C, H, W)
    return out

def truncate(x, thres=0.1):
    """Remove small values in heatmaps."""
    return torch.where(x < thres, torch.zeros_like(x), x)

def resize(x, p=2):
    """Resize heatmaps."""
    return x**p

def shift(x, N):
    """Shift N pixels up or down."""
    up = N >= 0
    N = abs(N)
    _, _, H, W = x.size()
    head = torch.arange(N)
    tail = torch.arange(H-N)

    if up:
        head = torch.arange(H-N)+N
        tail = torch.arange(N)
    else:
        head = torch.arange(N) + (H-N)
        tail = torch.arange(H-N)

    # permutation indices
    perm = torch.cat([head, tail]).to(x.device)
    out = x[:, :, perm, :]
    return out

def get_preds_fromhm(hm):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)
    return preds

def curve_fill(points, heatmapSize=256, sigma=3, erode=False):
    sigma = max(1,(sigma // 2)*2 + 1)
    points = points.astype(np.int32)
    canvas = np.zeros([heatmapSize, heatmapSize])
    cv2.fillPoly(canvas,np.array([points]),255)
    canvas = cv2.GaussianBlur(canvas, (sigma, sigma), sigma)
    return canvas.astype(np.float64)/255.0

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(64, 64, True, True, 256, first_one,
                                     out_channels=256,
                                     kernel_size=1, stride=1, padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))
        self.add_module('b2_' + str(level), ConvBlock(256, 256))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))
        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)
        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


class AddCoordsTh(nn.Module):
    def __init__(self, height=64, width=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            x_coords = torch.arange(height).unsqueeze(1).expand(height, width).float()
            y_coords = torch.arange(width).unsqueeze(0).expand(height, width).float()
            x_coords = (x_coords / (height - 1)) * 2 - 1
            y_coords = (y_coords / (width - 1)) * 2 - 1
            coords = torch.stack([x_coords, y_coords], dim=0)

            if self.with_r:
                rr = torch.sqrt(torch.pow(x_coords, 2) + torch.pow(y_coords, 2))
                rr = (rr / torch.max(rr)).unsqueeze(0)
                coords = torch.cat([coords, rr], dim=0)

            self.coords = coords.unsqueeze(0).to(device)
            self.x_coords = x_coords.to(device)
            self.y_coords = y_coords.to(device)

    def forward(self, x, heatmap=None):
        """
        x: (batch, c, x_dim, y_dim)
        """
        coords = self.coords.repeat(x.size(0), 1, 1, 1)

        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = torch.zeros_like(self.x_coords)
            xx_boundary_channel = torch.where(boundary_channel > 0.05, self.x_coords, zero_tensor).to(zero_tensor.device)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, self.y_coords, zero_tensor).to(zero_tensor.device)
            coords = torch.cat([coords, xx_boundary_channel, yy_boundary_channel], dim=1)

        x_and_coords = torch.cat([x, coords], dim=1)
        return x_and_coords


class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, height, width, with_r, with_boundary,
                 in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(height, width, with_r, with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        self.downsample = None
        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.BatchNorm2d(in_planes),
                                            nn.ReLU(True),
                                            nn.Conv2d(in_planes, out_planes, 1, 1, bias=False))

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


class FAN(nn.Module):
    def __init__(self, num_modules=1, end_relu=False, num_landmarks=98, fname_pretrained=None):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.end_relu = end_relu

        # Base part
        self.conv1 = CoordConvTh(256, 256, True, False,
                                 in_channels=3, out_channels=64,
                                 kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        self.add_module('m0', HourGlass(1, 4, 256, first_one=True))
        self.add_module('top_m_0', ConvBlock(256, 256))
        self.add_module('conv_last0', nn.Conv2d(256, 256, 1, 1, 0))
        self.add_module('bn_end0', nn.BatchNorm2d(256))
        self.add_module('l0', nn.Conv2d(256, num_landmarks+1, 1, 1, 0))

        if fname_pretrained is not None:
            self.load_pretrained_weights(fname_pretrained)

    def load_pretrained_weights(self, fname):
        if torch.cuda.is_available():
            checkpoint = torch.load(fname)
        else:
            checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        model_weights = self.state_dict()
        model_weights.update({k: v for k, v in checkpoint['state_dict'].items()
                              if k in model_weights})
        self.load_state_dict(model_weights)

    def forward(self, x):
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []
        boundary_channels = []
        tmp_out = None
        ll, boundary_channel = self._modules['m0'](x, tmp_out)
        ll = self._modules['top_m_0'](ll)
        ll = F.relu(self._modules['bn_end0']
                    (self._modules['conv_last0'](ll)), True)

        # Predict heatmaps
        tmp_out = self._modules['l0'](ll)
        if self.end_relu:
            tmp_out = F.relu(tmp_out)  # HACK: Added relu
        outputs.append(tmp_out)
        boundary_channels.append(boundary_channel)
        return outputs, boundary_channels

    @torch.no_grad()
    def get_heatmap(self, x):
        ''' outputs 0-1 normalized heatmap '''
        x = F.interpolate(x, size=256, mode='bilinear',align_corners=True)
        x_01 = x*0.5 + 0.5
        outputs, _ = self(x_01)
        heatmaps = outputs[-1][:, :-1, :, :]
        scale_factor = x.size(2) // heatmaps.size(2)
        heatmaps = F.interpolate(heatmaps, scale_factor=scale_factor,
                                     mode='bilinear', align_corners=True)
        return heatmaps


    @torch.no_grad()
    def get_points2heatmap(self, x):
        ''' outputs landmarks of x.shape '''
        heatmaps = self.get_heatmap(x)
        landmarks = []
        for i in range(x.size(0)):
            pred_landmarks = get_preds_fromhm(heatmaps[i].cpu().unsqueeze(0))
            landmarks.append(pred_landmarks)
        scale_factor = x.size(2) // heatmaps.size(2)
        landmarks = torch.cat(landmarks) * scale_factor
        heatmap_all=torch.zeros((len(x),7,x.size(2),x.size(2)))
        for i in range(0,len(x)):
            curves, boundary = self.points2curves(landmarks[i])
            heatmap = self.curves2segments(curves)
            heatmap = torch.from_numpy(heatmap).float()
            heatmap_all[i] = heatmap
        heatmap_all = heatmap_all.to(device=x.device)
        return heatmap_all, curves, boundary

    @torch.no_grad()
    def get_convex_hull(self, x):
        ''' outputs landmarks of x.shape '''
        heatmaps = self.get_heatmap(x)
        skins = []
        for i in range(x.size(0)):
            pred_landmark = get_preds_fromhm(heatmaps[i].cpu().unsqueeze(0))
            scale_factor = x.size(2) // heatmaps.size(2)
            pred_landmark = torch.squeeze(pred_landmark) * scale_factor
            curves_ref, _ = self.points2curves(pred_landmark)
            roi_ref = self.curves2segments(curves_ref)
            skin = roi_ref[0]
            skin = (skin > 0).astype(int)
            skin = 1- skin
            skin = torch.from_numpy(skin).type(dtype=torch.float32)
            skin = skin.unsqueeze(0)
            skin = skin.to(x.device)
            skins.append(skin)
        skins = torch.cat(skins)
        skins = skins.unsqueeze(1)
        return skins

    @torch.no_grad()
    def points2curves(self, points, heatmapSize=256, sigma=1,heatmap_num=15):
        curves = [0] * heatmap_num
        curves[0] = np.zeros((33, 2))  # contour
        curves[1] = np.zeros((5, 2))  # left top eyebrow
        curves[2] = np.zeros((5, 2))  # right top eyebrow
        curves[3] = np.zeros((4, 2))  # nose bridge
        curves[4] = np.zeros((5, 2))  # nose tip
        curves[5] = np.zeros((5, 2))  # left  bottom eye
        curves[6] = np.zeros((5, 2))  # left bottom eye
        curves[7] = np.zeros((5, 2))  # right top eye
        curves[8] = np.zeros((5, 2))  # right bottom eye
        curves[9] = np.zeros((7, 2))  # up up lip
        curves[10] = np.zeros((5, 2))  # up bottom lip
        curves[11] = np.zeros((5, 2))  # bottom up lip
        curves[12] = np.zeros((7, 2))  # bottom bottom lip
        curves[13] = np.zeros((5, 2))  # left bottom eyebrow
        curves[14] = np.zeros((5, 2))  # left bottom eyebrow
        # assignment proccess
        # countour
        for i in range(33):
            curves[0][i] = points[i,:]
        for i in range(5):
            # left top eyebrow
            curves[1][i][0] = points[i + 33, 0] - 10
            curves[1][i][1] = points[i + 33, 1]-40
            curves[2][i][0] = points[i + 42, 0] + 10
            curves[2][i][1] = points[i + 42, 1]-40
        # nose bridge
        for i in range(4):
            curves[3][i] = points[i + 51,:]
        # nose tip
        for i in range(5):
            curves[4][i] = points[i + 55,:]
        # left top eye
        for i in range(5):
            curves[5][i] = points[i + 60,:]
        # left bottom eye
        curves[6][0] = points[64,:]
        curves[6][1] = points[65,:]
        curves[6][2] = points[66,:]
        curves[6][3] = points[67,:]
        curves[6][4] = points[60,:]
        # right top eye
        for i in range(5):
            curves[7][i] = points[i + 68,:]
        # right bottom eye
        curves[8][0] = points[72,:]
        curves[8][1] = points[73,:]
        curves[8][2] = points[74,:]
        curves[8][3] = points[75,:]
        curves[8][4] = points[68,:]
        # up up lip
        for i in range(7):
            curves[9][i] = points[i + 76,:]
        # up bottom lip
        for i in range(5):
            curves[10][i] = points[i + 88,:]
        # bottom up lip
        curves[11][0] = points[92,:]
        curves[11][1] = points[93,:]
        curves[11][2] = points[94,:]
        curves[11][3] = points[95,:]
        curves[11][4] = points[88,:]
        # bottom bottom lip
        curves[12][0] = points[82,:]
        curves[12][1] = points[83,:]
        curves[12][2] = points[84,:]
        curves[12][3] = points[85,:]
        curves[12][4] = points[86,:]
        curves[12][5] = points[87,:]
        curves[12][6] = points[76,:]
        # left bottom eyebrow
        curves[13][0] = points[38,:]
        curves[13][1] = points[39,:]
        curves[13][2] = points[40,:]
        curves[13][3] = points[41,:]
        curves[13][4] = points[33,:]
        # right bottom eyebrow
        curves[14][0] = points[46,:]
        curves[14][1] = points[47,:]
        curves[14][2] = points[48,:]
        curves[14][3] = points[49,:]
        curves[14][4] = points[50,:]
        return curves, None

    @torch.no_grad()
    def curves2segments(self,  curves, heatmapSize=256, sigma=3):
        face = curve_fill(np.vstack([curves[0], curves[2][::-1], curves[1][::-1]]), heatmapSize, sigma)
        browL = curve_fill(np.vstack([curves[1], curves[13][::-1]]), heatmapSize, sigma)
        browR = curve_fill(np.vstack([curves[2], curves[14][::-1]]), heatmapSize, sigma)
        eyeL = curve_fill(np.vstack([curves[5], curves[6]]), heatmapSize, sigma)
        eyeR = curve_fill(np.vstack([curves[7], curves[8]]), heatmapSize, sigma)
        eye = np.max([eyeL, eyeR], axis=0)
        brow = np.max([browL, browR], axis=0)
        nose = curve_fill(np.vstack([curves[3][0:1], curves[4]]), heatmapSize, sigma)
        lipU = curve_fill(np.vstack([curves[9], curves[10][::-1]]), heatmapSize, sigma)
        lipD = curve_fill(np.vstack([curves[11], curves[12][::-1]]), heatmapSize, sigma)
        tooth = curve_fill(np.vstack([curves[10], curves[11][::-1]]), heatmapSize, sigma)
        return np.stack([face, brow, eye, nose, lipU, lipD, tooth])
    
    @torch.no_grad()
    def get_landmark_curve(self, x):  
        heatmaps = self.get_heatmap(x)
        landmarks = []
        for i in range(x.size(0)):
            pred_landmarks = get_preds_fromhm(heatmaps[i].cpu().unsqueeze(0))
            landmarks.append(pred_landmarks)
        scale_factor = x.size(2) // heatmaps.size(2)
        landmarks = torch.cat(landmarks) * scale_factor
        batch_landmark_figure =[]
        for i in range(0,len(x)):
            dpi = 100
            input = x[i] 
            preds = landmarks[i]
            fig = plt.figure(figsize=(input.shape[2] / dpi, input.shape[1] / dpi), dpi=dpi)
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(np.ones((input.shape[1],input.shape[2],input.shape[0])))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # eye
            ax.plot(preds[60:68,0],preds[60:68,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            ax.plot(preds[68:76,0],preds[68:76,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            #outer and inner lip
            ax.plot(preds[76:88,0],preds[76:88,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
            ax.plot(preds[88:96,0],preds[88:96,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            ax.axis('off')
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            batch_landmark_figure.append(data)
            plt.close(fig)
        return  batch_landmark_figure


