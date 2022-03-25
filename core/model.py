"""
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math
from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.wing import FAN



class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, id_dim, num_features):
        super().__init__()
        input_nc = 3
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(id_dim, num_features*2)
        self.conv_weight = nn.Conv2d(input_nc, num_features, kernel_size=3, padding=1)
        self.conv_bias = nn.Conv2d(input_nc, num_features, kernel_size=3, padding=1)

    def forward(self, x, s, mask, landmark):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        face_part = (1-mask[x.size(2)]) * x # face area; 
        norm_face_part = (1 + gamma) * self.norm(face_part) + beta
        landmark = F.interpolate(landmark, x.size(2), mode='bilinear',align_corners=True)
        weight_norm = self.conv_weight(landmark)
        bias_norm = self.conv_bias(landmark)
        norm_face_part = norm_face_part * (1+weight_norm) + bias_norm
        new_face = mask[x.size(2)] * x + (1-mask[x.size(2)])*norm_face_part
        return new_face


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, id_dim=512, 
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, id_dim)

    def _build_weights(self, dim_in, dim_out, id_dim):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(id_dim, dim_in)
        self.norm2 = AdaIN(id_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s, mask, landmark):
        x = self.norm1(x, s, mask, landmark)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s, mask,landmark)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s, mask,landmark):
        out = self._residual(x, s, mask,landmark)
        return out

class Generator(nn.Module):
    def __init__(self, img_size=256, id_dim=512, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.id_encoder   = IdentityEncoder(self.img_size, id_dim, max_conv_dim)
        self.attr_encoder = AttrEncoder(self.img_size,max_conv_dim)
        self.org_decoder  = Decoder(self.img_size,id_dim,max_conv_dim)
    def forward(self, x_a, x_b, x_a_lm, x_b_lm, x_a_mask=None, x_b_mask=None):
        x_a_attr, x_a_idvec, x_a_cache = self.encode(x_a, x_a_mask)
        x_b_attr, x_b_idvec, x_b_cache = self.encode(x_b, x_b_mask)
        x_ba, ms_features_ba, ms_outputs_ba = self.decode(x_b_attr, x_a_idvec, x_b_lm, x_b_cache, x_b_mask) # a's identity
        x_ab, ms_features_ab, ms_outputs_ab = self.decode(x_a_attr, x_b_idvec, x_a_lm, x_a_cache, x_a_mask) # b's identity
        return x_ba, x_ab, ms_features_ba, ms_features_ab, ms_outputs_ba, ms_outputs_ab
    def encode(self, image,mask=None):
        # encode an image to its attribute code and identity code
        id_vec, id_all_features = self.id_encoder(image)
        attr, attr_all_features, cache = self.attr_encoder(image, mask)
        return attr, id_vec, cache
    def decode(self, attr, id_vec, lm_image, cache, mask=None):
        image, ms_features, ms_outputs = self.org_decoder(attr, id_vec, lm_image, cache, mask)
        return image, ms_features, ms_outputs
    def encode_features(self, image, mask=None):
        # encode an image to its multiscale attribute feature and identity feature
        id_vec, id_all_features = self.id_encoder(image)
        attr, attr_all_features, cache = self.attr_encoder(image, mask)
        return attr_all_features, id_all_features


# attribute encoder
class AttrEncoder(nn.Module):
    def __init__(self, img_size=256, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 4
        repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, normalize=True, downsample=True)]
            dim_in = dim_out
        # bottleneck blocks
        for _ in range(2):
            blocks += [ResBlk(dim_out, dim_out, normalize=True)]
        self.model = nn.Sequential(*blocks)
    def forward(self, x, masks=None):
        attr_all_features = []
        cache = {}
        for block in self.model:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
            attr_all_features.append(x)
        return x, attr_all_features, cache

# identity encoder
class IdentityEncoder(nn.Module):
    def __init__(self, img_size=256, id_dim=512, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.model = nn.Sequential(*blocks)
        self.final_layer = nn.ModuleList()
        self.final_layer += [nn.Linear(dim_out, id_dim)]
    def forward(self, x):
        id_all_features = []
        for block in self.model:
            x = block(x)
            id_all_features.append(x)
        x = x.view(x.size(0), -1)  # batch_size, dim_out
        for block in self.final_layer:
            x = block(x)
            x = x.view(x.size(0), -1)
            id_all_features.append(x)
        return x, id_all_features

# decoder
class Decoder(nn.Module):
    def __init__(self, img_size=256, id_dim=512, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        dim_in_org = dim_in
        self.mask_size = [32,64,128]
        self.x_size =[8,16,32,64,128,256]
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))
        blocks = []
        repeat_num = int(np.log2(img_size)) - 4
        repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks.insert(0, AdainResBlk(dim_out, dim_in, id_dim,
                               upsample=True))
            dim_in = dim_out
        # bottleneck blocks
        for _ in range(2):
            blocks.insert(0, AdainResBlk(dim_out, dim_out, id_dim))
        self.model = nn.Sequential(*blocks)
        def to_rgb_output(dim_before_RGB):
            output = nn.Sequential(
            nn.InstanceNorm2d(dim_before_RGB, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_before_RGB, 3, 1, 1, 0))
            return output
        def skip_connection(dim_in_skip):
            dim_out_skip = dim_in_skip
            output = nn.Sequential(
            nn.Conv2d(dim_in_skip, dim_out_skip, 1, 1, 0))
            return output
        self.rgb_converters = nn.ModuleList()  
        self.skip_connects = nn.ModuleList()
        for i in self.mask_size:
            self.rgb_converters.append(to_rgb_output(int(dim_in_org*img_size/i)))
            self.skip_connects.append(skip_connection(int(dim_in_org*img_size/i)))

    def forward(self, x, id_vec,  lm_image, cache=None, mask=None):
        ms_outputs=[]
        ms_features=[]
        dict_masks={}
        if (mask is not None):
            for i in self.x_size:
                mask = F.interpolate(mask, i, mode='bilinear', align_corners=True)
                dict_masks[i] = mask
        index=0
        for block in self.model:
            mask = dict_masks[x.size(2)]
            x = block(x, id_vec, dict_masks, lm_image)  # 1-masks, face area
            if (mask is not None) and (x.size(2) in self.mask_size):
                mask = dict_masks[x.size(2)]
                x = x + self.skip_connects[index](mask * cache[x.size(2)])  # this mask should be attr
                ms_outputs.append(self.rgb_converters[index](x))
                index = index+1
            ms_features.append(x)
        x = self.to_rgb(x)
        return x, ms_features, ms_outputs


class Discriminator(nn.Module):
    def __init__(self, img_size=256, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        num_domains = 1
        blocks = []
        blocks += [nn.Conv2d(6, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x,lm_images):
        #out = self.main(x)
        x = torch.cat((x,lm_images), dim=-3)
        for block in self.main:
            x = block(x)
        out = x
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out


def build_model(config):
    generator = Generator(config['img_size'], config['id_dim'], max_conv_dim=512)
    discriminator = Discriminator(config['img_size'],max_conv_dim=512)
    generator_ema = copy.deepcopy(generator)
    nets = Munch(generator=generator,discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema)
    nets.fan  = FAN(fname_pretrained=config['wing_path']).eval()
    nets_ema.fan = nets.fan
    return nets, nets_ema