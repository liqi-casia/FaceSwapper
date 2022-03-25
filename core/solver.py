"""
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboardX
from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from core.face_model import Backbone



class Solver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets, self.nets_ema = build_model(config)
        self.arcface = Backbone(50, 0.6, 'ir_se')  # .to(device)
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load(config['face_model_path']))  # , strict=False
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)
        if config['mode'] == 'train':
            print(config)
            beta1 = config['beta1']
            beta2 = config['beta2']
            dis_params = list(self.nets['discriminator'].parameters())
            id_params = list(self.nets['generator'].id_encoder.parameters())
            dict_id_params = list(map(id, self.nets['generator'].id_encoder.parameters()))  # map is a function
            gen_params_wo_id = filter(lambda x: id(x) not in dict_id_params, self.nets['generator'].parameters())
            gen_id_params = []
            for p in id_params:
                if p.requires_grad:
                    gen_id_params.append(p)
            gen_params = []
            for p in gen_params_wo_id:
                if p.requires_grad:
                    gen_params.append(p)

            for net in self.nets.keys():
                if net == 'generator':
                    self.gen_opt = torch.optim.Adam([{'params': gen_params},
                        {'params': gen_id_params,'lr':config['id_lr']}],
                        lr=config['lr'],
                        betas=[beta1, beta2],
                        weight_decay=config['weight_decay'])
                elif net == 'discriminator':
                    self.dis_opt = torch.optim.Adam(
                        [p for p in dis_params if p.requires_grad],
                        lr=config['lr'],
                        betas=[beta1, beta2],
                        weight_decay=config['weight_decay'])
            self.optims = Munch()
            self.optims['generator'] = self.gen_opt
            self.optims['discriminator'] = self.dis_opt
            self.ckptios = [
                CheckpointIO(ospj(config['checkpoint_dir'], '{:06d}_nets.ckpt'), **self.nets),
                CheckpointIO(ospj(config['checkpoint_dir'], '{:06d}_nets_ema.ckpt'), **self.nets_ema),
                CheckpointIO(ospj(config['checkpoint_dir'], '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(config['test_checkpoint_dir'], **self.nets_ema)]
        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the pretrained network parameters
            if ('ema' not in name) and ('fan' not in name) and ('arcface' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)
        # Setup logger and output folders
        model_name = config['dataset']
        timestamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
        self.train_writer = tensorboardX.SummaryWriter(os.path.join(config['log_dir'] + model_name, timestamp))
    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)
    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)
    def _load_test_checkpoint(self, ckptname):
        for ckptio in self.ckptios:
            ckptio.load_test(ckptname)
    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        config = self.config
        nets = self.nets
        nets_ema = self.nets_ema
        gen_opt = self.gen_opt
        dis_opt = self.dis_opt
        arcface = self.arcface
        fetcher = InputFetcher(loaders.src,  'train')
        inputs_val = next(fetcher)
        # resume training if necessary
        if config['resume_iter'] > 0:
            self._load_checkpoint(config['resume_iter'])
        print('Start training...')
        start_time = time.time()
        for i in range(config['resume_iter'], config['total_iters']):
            # fetch images 
            inputs = next(fetcher)
            src, tar, src_lm, tar_lm, src_mask, tar_mask= inputs.src, inputs.tar, inputs.src_lm, \
                                              inputs.tar_lm, inputs.src_mask, inputs.tar_mask
            # train the discriminator
            d_loss, d_losses_all = compute_d_loss(
                nets, config, src,  tar, src_lm, tar_lm, src_mask, tar_mask)
            self._reset_grad()
            d_loss.backward()
            dis_opt.step()
            # train the generator
            g_loss, g_losses_all = compute_g_loss(
                nets, config, src,  tar, src_lm, tar_lm, src_mask, tar_mask, arcface)
            self._reset_grad()
            g_loss.backward()
            gen_opt.step()
            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            if (i+1) % config['print_every'] == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, config['total_iters'])
                all_losses = dict()
                for loss, prefix in zip([d_losses_all, g_losses_all],
                                        ['D/all_',  'G/all_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                        self.train_writer.add_scalar(prefix+key, value, i+1)
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
            # generate images for observation
            if (i+1) % config['sample_every'] == 0:
                os.makedirs(config['sample_dir'], exist_ok=True)
                utils.display_image(nets_ema, config, inputs=inputs_val, step=i+1)
            # save model checkpoints
            if (i+1) % config['save_every'] == 0:
                self._save_checkpoint(step=i+1)
        self.train_writer.close()

    @torch.no_grad()
    def test(self, loaders):
        config = self.config
        nets_ema = self.nets_ema
        os.makedirs(config['result_dir'], exist_ok=True)
        self._load_test_checkpoint(config['test_checkpoint_name'])
        f = open(config['test_img_list'], 'r')
        img_num = len(f.readlines())
        f.close()
        total_iters = int(img_num/config['batch_size']) + 1
        save_dir=config['result_dir']
        test_fetcher = InputFetcher(loaders.src, 'test')
        for i in range(0, total_iters):
            inputs = next(test_fetcher)
            utils.disentangle_and_swapping_test(nets_ema, config, inputs, save_dir)

def compute_d_loss(nets, config, x_a, x_b, x_a_lm, x_b_lm, x_a_mask, x_b_mask):
    x_a.requires_grad_()
    x_b.requires_grad_()
    out_a = nets.discriminator(x_a,x_a_lm)
    out_b = nets.discriminator(x_b,x_b_lm)
    loss_real_a = adv_loss(out_a, 1)
    loss_real_b = adv_loss(out_b, 1)
    loss_reg_a = r1_reg(out_a, x_a)
    loss_reg_b = r1_reg(out_b, x_b)
    loss_real = loss_real_a + loss_real_b
    loss_reg = loss_reg_a + loss_reg_b
    x_ba, x_ab, ms_features_ba, ms_features_ab, ms_outputs_ba,\
        ms_outputs_ab = nets.generator(x_a, x_b, x_a_lm, x_b_lm, x_a_mask, x_b_mask)  
    out_ba = nets.discriminator(x_ba, x_b_lm)  # x_ba， a's identity
    out_ab = nets.discriminator(x_ab, x_a_lm)  # x_ab， b's identity
    loss_fake_ba = adv_loss(out_ba, 0)
    loss_fake_ab = adv_loss(out_ab, 0)
    loss_fake = loss_fake_ba + loss_fake_ab
    loss = loss_real + loss_fake + config['lambda_reg'] * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item(),
                       total_loss=loss.item())

def compute_g_loss(nets, config, x_a, x_b, x_a_lm, x_b_lm, x_a_mask, x_b_mask, arcface):
    att_a, i_a_prime, cache_a  = nets.generator.encode(x_a, x_a_mask) 
    att_b, i_b_prime, cache_b  = nets.generator.encode(x_b, x_b_mask)
    x_a_recon, ms_features_a, ms_outputs_a = nets.generator.decode(att_a, i_a_prime, x_a_lm, cache_a, x_a_mask)
    x_b_recon, ms_features_b, ms_outputs_b = nets.generator.decode(att_b, i_b_prime, x_b_lm, cache_b, x_b_mask)
    x_ba, ms_features_ba, ms_outputs_ba = nets.generator.decode(att_b, i_a_prime, x_b_lm, cache_b, x_b_mask)  # x_ba， a's identity
    x_ab, ms_features_ab, ms_outputs_ab = nets.generator.decode(att_a, i_b_prime, x_a_lm, cache_a, x_a_mask)  # x_ab， b's identity
    with torch.no_grad():
        a_embed, a_feats = arcface(F.interpolate(x_a, [112, 112], mode='bilinear', align_corners=True))
        b_embed, b_feats = arcface( F.interpolate(x_b, [112, 112], mode='bilinear', align_corners=True))
    ba_embed, ba_feats = arcface(F.interpolate(x_ba, [112, 112], mode='bilinear', align_corners=True))
    ab_embed, ab_feats = arcface(F.interpolate(x_ab, [112, 112], mode='bilinear', align_corners=True))
    loss_id_a = (1 - F.cosine_similarity(a_embed, ba_embed, dim=1))
    loss_id_b=(1 - F.cosine_similarity(b_embed, ab_embed, dim=1))
    loss_id_a = loss_id_a.mean()
    loss_id_b = loss_id_b.mean()
    loss_id = loss_id_a + loss_id_b
    out_a = nets.discriminator(x_ba, x_b_lm) #x_ba, a's identity
    out_b = nets.discriminator(x_ab, x_a_lm) #x_ab, b's identity
    loss_adv_a = adv_loss(out_a, 1)
    loss_adv_b = adv_loss(out_b, 1)
    loss_adv = loss_adv_a + loss_adv_b
    loss_recon_x_a = torch.mean(torch.abs(x_a_recon - x_a))
    loss_recon_x_b = torch.mean(torch.abs(x_b_recon - x_b))
    loss_recon = loss_recon_x_a + loss_recon_x_b
    loss_att_a_face = style_loss_face(ms_features_a, ms_features_ab, x_a_mask)
    loss_att_b_face = style_loss_face(ms_features_b, ms_features_ba, x_b_mask)
    loss_att_face   = loss_att_a_face + loss_att_b_face
    loss_att_a_bg = style_loss_background(ms_features_a, ms_features_ab,x_a_mask)
    loss_att_b_bg = style_loss_background(ms_features_b, ms_features_ba,x_b_mask)
    loss_att_bg = loss_att_a_bg + loss_att_b_bg
    loss = loss_adv + config['lambda_id'] * loss_id \
         + config['lambda_recon'] * loss_recon + config['lambda_att_face'] * loss_att_face \
           + config['lambda_att_bg'] * loss_att_bg
    return loss, Munch(adv=loss_adv.item(),
                       id=loss_id.item(),
                       recon=loss_recon.item(),
                       att_face=loss_att_face.item(),
                       att_bg=loss_att_bg.item(),
                       total_loss=loss.item())

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

def compute_gram(x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (h * w * ch)
    return G

def style_loss_face(x, y,masks):
    style_loss = 0.0
    for i in range(0, len(x)):
        masks = F.interpolate(masks, x[i].size(2), mode='bilinear')
        style_loss += torch.nn.L1Loss()(compute_gram((1-masks)*x[i]), compute_gram((1-masks)*y[i]))
    style_loss = style_loss/len(x)
    return style_loss

def style_loss_background(x, y, masks):
    style_loss = 0.0
    for i in range(0, len(x)):
        masks = F.interpolate(masks,x[i].size(2),mode='bilinear')
        style_loss += torch.nn.L1Loss()(masks*x[i], masks*y[i])  # pay attention to this form
    style_loss = style_loss/len(x)
    return style_loss







