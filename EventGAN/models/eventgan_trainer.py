from models.unet import UNet
import torch
from torchvision.utils import make_grid
from datasets import event_loader
from models import decoder
import utils.event_utils as event_utils
import torch.nn as nn
import torch.nn.functional as F
from losses import generator_loss, discriminator_loss, multi_scale_flow_loss
from utils.viz_utils import flow2rgb, gen_event_images
from utils.utils import get_latest_checkpoint

import radam
import pytorch_ssim
import pytorch_utils
from pytorch_utils import CheckpointDataLoader, CheckpointSaver
import numpy as np
import os

def build_gan(options):
    generator = UNet(num_input_channels=2*options.n_image_channels,
                     num_output_channels=options.n_time_bins * 2,
                     skip_type='concat',
                     activation='relu',
                     num_encoders=4,
                     base_num_channels=32,
                     num_residual_blocks=2,
                     norm='BN',
                     use_upsample_conv=True,
                     with_activation=True,
                     sn=options.sn,
                     multi=False)

    discriminator = decoder.Patch_Discriminator(
        event_channel=options.n_time_bins*2,
        image_channel=options.n_image_channels,
        ndf=options.num_filter_disc,
        n_layers=4
    )

    return generator, discriminator

class EventGANTrainer(pytorch_utils.BaseTrainer):
    def __init__(self, options, train=True):
        self.is_training = train
        super(EventGANTrainer, self).__init__(options)

    def init_fn(self):
        # build model
        self.generator, self.discriminator = build_gan(self.options)

        self.models_dict = {"gen": self.generator, 
                            "dis": self.discriminator}
        if not self.is_training:
            self.optimizers_dict = {}
            return
        
        if self.options.cycle_recons:
            model_folder = "EventGAN/pretrained_models/{}".format(self.options.cycle_recons_model)
            checkpoint = os.path.join(model_folder, os.listdir(model_folder)[-1])
            self.cycle_unet_recons = torch.load(checkpoint)
            self.cycle_unet_recons.eval()
            self.models_dict["e2i"] = self.cycle_unet_recons
        if self.options.cycle_flow:
            model_folder = "EventGAN/pretrained_models/{}".format(self.options.cycle_flow_model)
            checkpoint = os.path.join(model_folder, os.listdir(model_folder)[-1])
            self.cycle_unet_flow = torch.load(checkpoint)
            self.cycle_unet_flow.eval()
            self.models_dict["e2f"] = self.cycle_unet_flow

        # params for each part of the network
        dis_params = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        gen_params = filter(lambda p: p.requires_grad, self.generator.parameters())
        gen_params = self.generator.parameters()


        optimizer_dis = radam.RAdam(dis_params,
                                    lr=self.options.lrd,
                                    weight_decay=0.,
                                    betas=(0., 0.999))
        
        optimizer_gen = radam.RAdam(list(gen_params),
                                    lr=self.options.lrg,
                                    weight_decay=0.,
                                    betas=(0., 0.999))

        self.ssim = pytorch_ssim.SSIM()
        self.secondary_l1 = nn.L1Loss(reduction="mean")
        self.image_loss = lambda x, y: self.secondary_l1(x, y) - self.ssim(x, y)
        
        self.optimizers_dict = { "optimizer_gen" : optimizer_gen,
                                 "optimizer_dis" : optimizer_dis}
                    
        self.train_ds, self.train_sampler = event_loader.get_and_concat_datasets(
            self.options.train_file,
            self.options,
            train=True)
        self.validation_ds, self.validation_sampler = event_loader.get_and_concat_datasets(
            self.options.validation_file,
            self.options,
            train=False)
        
        self.cdl_kwargs["collate_fn"] = event_utils.none_safe_collate
        self.cdl_kwargs["sampler"] = self.train_sampler

        self.prev_gen_losses = {}
        self.prev_dis_losses = {}
        self.prev_gen_outputs = {}
        self.prev_dis_outputs = {}

    """
    Given a (potentially batched) pair of images, computes a forward pass of the network to 
    generate an event volume.

    Inputs: input_images - (B x ) 2 x H x W, stacked tensor of previous and next images along the
                           channel dimension. Batch dimension is optional.
    Outputs: gen_fake_volume - (B x ) (2 x n_time_bins) x H x W, 
                               output event volume from the network.
    """
    def forward_pass(self, input_images, is_train=False):
        if len(input_images.shape) == 3:
            input_images = input_images[None, ...]
        assert len(input_images.shape) == 4 and input_images.shape[1] == 2 , \
            "Input images to forward pass must be of shape 2 x H x W or B x 2 x H x W."
        gen_model = self.models_dict['gen']
        if not is_train:
            with torch.no_grad():
                gen_model.eval()
                gen_fake_volume = gen_model(input_images)
        else:
            gen_fake_volume = gen_model(input_images)
            
        return gen_fake_volume
        
    def train_step(self, input_batch):
        if not input_batch:
            return {}, {}

        # alternate between gen loss and dis loss
        mod_step = self.step_count % (self.options.disc_iter + self.options.gen_iter)
        if mod_step < self.options.disc_iter and not self.options.no_train_gan:
            optimizer_D = self.optimizers_dict['optimizer_dis']
            optimizer_D.zero_grad()
            d_final_loss, d_losses, d_outputs = self.discriminator_train_step(input_batch)
            d_final_loss.backward()
            optimizer_D.step()
            
            self.prev_dis_losses, self.prev_dis_outputs = d_losses, d_outputs
            return d_losses, d_outputs
        else:
            optimizer_G = self.optimizers_dict['optimizer_gen']
            optimizer_G.zero_grad()
            gen_prev_images = input_batch["prev_image"]
            gen_next_images = input_batch["next_image"]
            gen_prev_images_gt = input_batch["prev_image_gt"]
            gen_next_images_gt = input_batch["next_image_gt"]
            gen_images = torch.cat([gen_prev_images, gen_next_images], 1)
            gen_event_volume = input_batch["event_volume"]

            g_final_loss, g_losses, g_outputs = self.generator_train_step(gen_prev_images,
                                                                          gen_next_images,
                                                                          gen_prev_images_gt,
                                                                          gen_next_images_gt,
                                                                          gen_images,
                                                                          gen_event_volume)
            g_final_loss.backward()
                        
            optimizer_G.step()
            self.prev_gen_losses, self.prev_gen_outputs = g_losses, g_outputs
            return g_losses, g_outputs
            
    def generator_train_step(self,
                             gen_prev_images, gen_next_images,
                             gen_prev_images_gt, gen_next_images_gt,
                             gen_images, gen_event_volume):
        gen_model = self.models_dict['gen']
        dis_model = self.models_dict['dis']
        if self.options.cycle_recons:
            e2i_model = self.models_dict['e2i']
        if self.options.cycle_flow:
            e2f_model = self.models_dict['e2f']

        losses = {}
        outputs = {}
        g_loss = 0.
        # Train generator.
        # Generator output.
        gen_fake_volume = gen_model(gen_images)
        
        if not self.options.no_train_gan:
            # Get discriminator prediction.
            classification = dis_model(gen_fake_volume[::-1], gen_images)
            # Compute GAN loss.
            g_loss += generator_loss("hinge", classification)
            losses['generator'] = g_loss

        cycle_loss = 0.
        # cycle consistency loss.
        if self.options.cycle_recons:
            e2i_input = torch.sum(gen_fake_volume[-1], dim=1, keepdim=True)
            e2i_input = torch.cat([e2i_input, gen_prev_images], dim=1)
            recons_image_list = e2i_model(e2i_input)

            reconstruction_loss = [self.image_loss(F.interpolate(r_img, gen_next_images.shape[2:]),
                                                   gen_next_images) \
                                   for r_img in recons_image_list]
                
            reconstruction_loss = torch.sum(torch.stack(
                [loss * 2. ** (i - len(reconstruction_loss) + 1) \
                 for i, loss in enumerate(reconstruction_loss)]))
            #    recons_image,
            cycle_loss += self.options.cycle_recons_weight * reconstruction_loss
            losses['cycle_reconstruction_loss'] = reconstruction_loss
            outputs['reconstructed_image'] = (recons_image_list[-1] + 1.) / 2.
        if self.options.cycle_flow:
            flow_mask = torch.sum(gen_event_volume, 1) > 0
            e2f_input = gen_fake_volume[-1]
            flow_output = e2f_model(e2f_input)
            photo_loss, smooth_loss, _ = multi_scale_flow_loss(
                gen_prev_images_gt,
                gen_next_images_gt,
                flow_output,
                flow_mask,
                second_order_smooth=False)
            flow_loss = photo_loss
            if not self.options.no_flow_smoothness:
                flow_loss += smooth_loss * self.options.smooth_weight
            cycle_loss += self.options.cycle_flow_weight * flow_loss
            losses['cycle_flow_loss'] = flow_loss
            outputs['cycle_flow'] = flow2rgb(flow_output[-1])
        g_loss += cycle_loss
        
        # Other outputs for visualization.
        outputs.update(gen_event_images(gen_fake_volume[-1], 'gen',
                                        self.device, self.options.normalize_events))
        outputs.update(gen_event_images(gen_event_volume, 'raw',
                                        self.device, self.options.normalize_events))
        outputs['gt_gray'] = (gen_next_images + 1.) / 2.
        outputs['gen_event_hist'] = gen_fake_volume[-1]
        outputs['raw_event_hist'] = gen_event_volume
        return g_loss, losses, outputs

    def discriminator_train_step(self, batch):
        gen_model = self.models_dict['gen']
        dis_model = self.models_dict['dis']
        
        dis_prev_images = batch["prev_image"]
        dis_next_images = batch["next_image"]
        dis_prev_images_gt = batch["prev_image_gt"]
        dis_next_images_gt = batch["next_image_gt"]
        dis_images = torch.cat([dis_prev_images, dis_next_images], 1)
        dis_images_gt = torch.cat([dis_prev_images_gt, dis_next_images_gt], 1)
        dis_event_volume = batch["event_volume"]

        split_size = dis_images.shape[0] // 2
        
        # inference with generator
        # optimize the discriminator
        # First half of discriminator data is for fake.
        # Second half of discriminator data is for real.
        dis_images_fake = dis_images[:split_size, ...]
        dis_images_real = dis_images[split_size:, ...]
        dis_images_fake_gt = dis_images_gt[:split_size, ...]
        dis_images_real_gt = dis_images_gt[split_size:, ...]
        event_volume_real = dis_event_volume[split_size:, ...]

        dis_gen_fake_volume = gen_model(dis_images_fake)
        
        fake_logits = dis_model([dis_gen_fake_volume[-1]], dis_images_fake_gt)
        real_logits = dis_model([event_volume_real], dis_images_real_gt)

        # If rand_n is less than flip_label, flip labels.
        rand_n = np.random.random(real_logits.shape)
        do_flip = torch.from_numpy(
            np.greater(rand_n, self.options.flip_label).astype(np.uint8)).to(self.device).byte()

        real_logits_maybe_flipped = torch.where(do_flip,
                                                real_logits,
                                                fake_logits)
        fake_logits_maybe_flipped = torch.where(do_flip,
                                                fake_logits,
                                                real_logits)
        d_loss = discriminator_loss("hinge", real_logits_maybe_flipped, fake_logits_maybe_flipped)
            
        losses = {
            'discriminator' : d_loss,
            'fake_class_acc' : torch.mean(torch.eq(
                torch.sign(fake_logits),
                -torch.ones(fake_logits.size()).to(self.device)).float()),
            'real_class_acc' : torch.mean(torch.eq(
                torch.sign(real_logits),
                torch.ones(real_logits.size()).to(self.device)).float())
        }
        
        outputs = {}
        return d_loss, losses, outputs
    
    def summaries(self, input_batch, losses, output, mode="train"):
        nrow = 4
        self.summary_writer.add_scalar("{}/learning_rate".format(mode),
                                       self.get_lr(),
                                       self.step_count)
        for k, v in losses.items():
            self.summary_writer.add_scalar("{}/{}".format(mode, k), v, self.step_count)
        for k, v in output.items():
            if 'hist' in k:
                self.summary_writer.add_histogram("{}/{}".format(mode, k),
                                                  v, self.step_count)
            else:
                self.summary_writer.add_image("{}/{}".format(mode, k),
                                              make_grid(v, nrow=nrow), self.step_count)
                
    def train_summaries(self, input_batch, losses, output):
        self.summaries(input_batch,
                       {**self.prev_gen_losses, **self.prev_dis_losses},
                       {**self.prev_gen_outputs, **self.prev_dis_outputs},
                       mode="train")

    def test(self, subset_num=50):
        test_cdl = { k:v for k,v in self.cdl_kwargs.items() }
        test_cdl["sampler"] = self.validation_sampler
        test_data_loader = CheckpointDataLoader(self.validation_ds,
                                                **test_cdl)
        test_data_loader.next_epoch(None)

        cumulative_losses = {}
        i = 0
        with torch.no_grad():
            for model in self.models_dict:
                self.models_dict[model].eval()
            
            for step, batch in enumerate(test_data_loader):
                if i >= subset_num:
                    break
                batch = {k: v.to(self.device) for k,v in batch.items() }

                gen_prev_images = batch["prev_image"]
                gen_next_images = batch["next_image"]
                gen_prev_images_gt = batch["prev_image_gt"]
                gen_next_images_gt = batch["next_image_gt"]
                gen_images = torch.cat([gen_prev_images, gen_next_images], 1)
                gen_event_volume = batch["event_volume"]
                
                g_final_loss, g_losses, g_outputs = self.generator_train_step(gen_prev_images,
                                                                              gen_next_images,
                                                                              gen_prev_images_gt,
                                                                              gen_next_images_gt,
                                                                              gen_images,
                                                                              gen_event_volume)

                for k, v in g_losses.items():
                    if k in cumulative_losses:
                        cumulative_losses[k] += v
                    else:
                        cumulative_losses[k] = v
                i += 1

        cumulative_losses = { k:v/float(i) for k, v in cumulative_losses.items() }
                
        for model in self.models_dict:
            self.models_dict[model].train()
        self.summaries(batch, cumulative_losses, g_outputs, mode="test")
