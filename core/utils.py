"""
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml

def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print("Number of parameters of %s: %i" % (name, num_params))
def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)
def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream,Loader=yaml.FullLoader)


@torch.no_grad()
def disentangle_and_reconstruct(nets, config, src, tar, src_lm, tar_lm, src_mask, tar_mask, filename):
    N, C, H, W = src.size()
    att_a, i_a_prime, cache_a = nets.generator.encode(src, src_mask)  # cache follows attr
    att_b, i_b_prime, cache_b = nets.generator.encode(tar, tar_mask)
    a_recon,_,_ = nets.generator.decode(att_a, i_a_prime, src_lm, cache_a, src_mask)
    b_recon,_,_ = nets.generator.decode(att_b, i_b_prime, tar_lm, cache_b, tar_mask)
    disp_concat = [src, tar, a_recon, b_recon]
    disp_concat = torch.cat(disp_concat, dim=0)
    save_image(disp_concat, N, filename)

@torch.no_grad()
def disentangle_and_swapping(nets, config, src, tar, src_lm, tar_lm, src_mask, tar_mask, filename):
    N, C, H, W = src.size()
    whitespace = torch.ones(1, C, H, W).to(src.device)
    src_with_whitespace = torch.cat([whitespace, src], dim=0)
    disp_concat = [src_with_whitespace]
    for i, tar_img in enumerate(tar):
        tar_imgs = tar_img.repeat(N,1,1,1)
        disp_img = tar_img.repeat(1,1,1,1)
        tar_i_lm = tar_lm[i,:,:,:]
        tar_i_lm = tar_i_lm.repeat(N,1,1,1)
        tar_i_mask = tar_mask[i,:,:,:]
        tar_i_mask = tar_i_mask.repeat(N,1,1,1)
        srcid_taratt, tarid_srcatt,_,_,_,_ = nets.generator(src, tar_imgs, src_lm,
                                                                tar_i_lm, src_mask, tar_i_mask)
        fake_srcid_taratt = torch.cat([disp_img, srcid_taratt], dim=0)
        fake_tarid_srcatt = torch.cat([disp_img, tarid_srcatt], dim=0)
        disp_concat += [fake_srcid_taratt]
        disp_concat += [fake_tarid_srcatt]
    disp_concat = torch.cat(disp_concat, dim=0)
    save_image(disp_concat, N+1, filename)

@torch.no_grad()
def display_image(nets, config, inputs, step):
    src, tar, src_lm, tar_lm, src_mask, tar_mask = inputs.src, inputs.tar, \
                                       inputs.src_lm, inputs.tar_lm, inputs.src_mask, inputs.tar_mask
    # face reconstruction 
    filename = ospj(config['sample_dir'], '%06d_reconstruction.jpg' % (step))
    disentangle_and_reconstruct(nets, config, src, tar, src_lm, tar_lm, src_mask, tar_mask, filename)
    # face swapping
    filename = ospj(config['sample_dir'], '%06d_faceswapping.jpg' % (step))
    disentangle_and_swapping(nets, config, src, tar, src_lm, tar_lm, src_mask, tar_mask, filename)

@torch.no_grad()
def disentangle_and_swapping_test(nets, config, inputs, save_dir):
    post_process = config['post_process']
    src, tar, src_lm, tar_lm, src_mask, tar_mask, tar_parsing, src_name, tar_name = inputs.src, inputs.tar, \
            inputs.src_lm, inputs.tar_lm, inputs.src_mask, inputs.tar_mask, inputs.tar_parsing, inputs.src_name,  inputs.tar_name,
    src_mask = F.interpolate(src_mask, src.size(2), mode='bilinear', align_corners=True)
    tar_mask = F.interpolate(tar_mask, src.size(2), mode='bilinear', align_corners=True)
    srcid_taratt, tarid_srcatt,_,_,_,_ = nets.generator(src, tar, src_lm, tar_lm, src_mask, tar_mask) #modified by liqi
    result_first  = save_dir + 'swapped_result_single/'
    result_second = save_dir + 'swapped_result_afterps/'
    result_third  = save_dir + 'swapped_result_all/'
    if not os.path.exists(result_first):
        os.makedirs(result_first)
    if not os.path.exists(result_second):
        os.makedirs(result_second)
    if not os.path.exists(result_third):
        os.makedirs(result_third)
    if post_process:
        src_convex_hull = nets.fan.get_convex_hull(src)
        tar_convex_hull = nets.fan.get_convex_hull(tar)
        temp_src_forehead = src_convex_hull - src_mask
        temp_tar_forehead = tar_convex_hull - tar_mask
        # to ensure the values of src_forehead and tar_forehead are in [0,1]
        one_tensor  = torch.ones(temp_src_forehead.size()).to(device=temp_src_forehead.device)
        zero_tensor = torch.zeros(temp_src_forehead.size()).to(device=temp_src_forehead.device)
        temp_var = torch.where(temp_src_forehead >= 1.0, one_tensor, temp_src_forehead)
        src_forehead = torch.where(temp_var  <= 0.0, zero_tensor, temp_var)
        temp_var = torch.where(temp_tar_forehead >= 1.0, one_tensor, temp_tar_forehead)
        tar_forehead = torch.where(temp_var <= 0.0, zero_tensor, temp_var)
        tar_hair = get_hair(tar_parsing)
        post_result = postprocess(tar, srcid_taratt, tar_hair, src_forehead, tar_forehead)
    for i in range(len(srcid_taratt)):
        filename = result_first + src_name[i][0:-4]+'_FS_'+ tar_name[i][0:-5]+'.png'
        filename_post = result_second + src_name[i][0:-4] + '_FS_' + tar_name[i][0:-5] + '.png'
        filename_all  = result_third + src_name[i][0:-4] + '_FS_' + tar_name[i][0:-5] + '.png'
        save_image(srcid_taratt[i,:,:,:], 1, filename)
        if post_process:
            save_image(post_result[i, :, :, :], 1, filename_post)
            x_concat = torch.cat([src[i].unsqueeze(0), tar[i].unsqueeze(0),
                                srcid_taratt[i, :, :, :].unsqueeze(0),post_result[i, :, :, :].unsqueeze(0)], dim=0)
            save_image(x_concat, 4, filename_all)   
        else:
            x_concat = torch.cat([src[i].unsqueeze(0), tar[i].unsqueeze(0),
                                srcid_taratt[i, :, :, :].unsqueeze(0)], dim=0)
            save_image(x_concat, 3, filename_all)


def get_hair(segmentation):
    out = segmentation.mul_(255).int()
    mask_ind_hair = [17]
    with torch.no_grad():
        out_parse = out
        hair = torch.ones((out_parse.shape[0], 1, out_parse.shape[2], out_parse.shape[3])).cuda()
        for pi in mask_ind_hair:
            index = torch.where(out_parse == pi)
            hair[index[0], :, index[2],index[3]] = 0
    return  hair


def postprocess(tar, srcid_taratt, tar_hair, src_forehead, tar_forehead):
    #inner area of tar_hair is  0, inner area of tar_forehead is  1
    smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
    one_tensor = torch.ones(tar_forehead.size()).to(device=tar_forehead.device)
    temp_tar_hair_and_forehead = (1-tar_hair) + tar_forehead
    tar_hair_and_forehead = torch.where(temp_tar_hair_and_forehead >= 1.0, one_tensor, temp_tar_hair_and_forehead)
    tar_preserve = tar_hair
    # find whether occlusion exists in source image; if exists, then preserve the hair and forehead of the target image
    for i in range(src_forehead.size(0)):
        src_forehead_i = src_forehead[i,:,:,:]
        src_forehead_i  = src_forehead_i .squeeze_()
        tar_forehead_i = tar_forehead[i,:,:,:]
        tar_forehead_i = tar_forehead_i.squeeze_()
        H1,W1 = torch.nonzero(src_forehead_i).size()
        H2,W2 = torch.nonzero(tar_forehead_i).size()
        if (H1 * W1) / (H2 * W2 + 0.0001) < 0.4 and (H2 * W2) >= 1000:  #
            tar_preserve[i,:,:,:] = 1 - tar_hair_and_forehead[i,:,:,:]
    soft_mask, _ = smooth_mask(tar_preserve)
    result =  srcid_taratt * soft_mask + tar * (1-soft_mask)
    return result


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold
        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)
        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()
        return x, mask