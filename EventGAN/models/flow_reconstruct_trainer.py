import torch
import torch.nn as nn
from torchvision.utils import make_grid

from datasets import event_loader
from models.unet import UNet
from losses import multi_scale_flow_loss

import utils.event_utils as event_utils
from utils.viz_utils import flow2rgb

import pytorch_ssim
import radam

import pytorch_utils
from pytorch_utils import CheckpointDataLoader, CheckpointSaver

class FlowReconstructTrainer(pytorch_utils.BaseTrainer):
    def __init__(self, options):
        super(FlowReconstructTrainer, self).__init__(options)

    def init_fn(self):
        if self.options.model == 'flow':
            num_input_channels = self.options.n_time_bins * 2
            num_output_channels = 2
        elif self.options.model == 'recons':
            # For the reconstruction model, we sum the event volume across the time dimension, so
            # that the network only sees a single channel event input, plus the prev image.
            num_input_channels = 1 + self.options.n_image_channels
            num_output_channels = self.options.n_image_channels
        else:
            raise ValueError("Class was initialized with an invalid model {}"
                             ", only {EventGAN, flow, recons} are supported."
                             .format(self.options.model))
        
        self.cycle_unet = UNet(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            skip_type='concat',
            activation='tanh',
            num_encoders=4,
            base_num_channels=32,
            num_residual_blocks=2,
            norm='BN',
            use_upsample_conv=True,
            multi=True)

        self.models_dict = { "model": self.cycle_unet }
        model_params = self.cycle_unet.parameters()

        optimizer = radam.RAdam(list(model_params),
                                lr = self.options.lrc,
                                weight_decay=self.options.wd,
                                betas = (self.options.lr_decay, 0.999))

        self.ssim = pytorch_ssim.SSIM()
        self.l1 = nn.L1Loss(reduction="mean")
        self.image_loss = lambda x, y: self.l1(x, y) - self.ssim(x, y)

        self.optimizers_dict = {"optimizer" : optimizer }

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

    """
    Main training loop iteration.
    """
    def train_step(self, input_batch):
        if not input_batch:
            return {}, {}

        optimizer = self.optimizers_dict['optimizer']
        model = self.models_dict['model']

        # Get images and events.
        prev_image = input_batch["prev_image"]
        next_image = input_batch["next_image"]
        prev_image_gt = input_batch["prev_image_gt"]
        next_image_gt = input_batch["next_image_gt"]
        
        event_volume = input_batch["event_volume"]

        losses = {}
        outputs = {}

        optimizer.zero_grad()
        loss = 0.

        # Generate network input.
        if self.options.model == 'flow':
            network_input = event_volume
        elif self.options.model == 'recons':
            event_image = torch.sum(event_volume, 1, keepdim=True)
            network_input = torch.cat([event_image, prev_image], 1)

        network_output = model(network_input)

        # Compute losses.
        if self.options.model == 'flow':
            flow_mask = torch.sum(event_volume, 1) > 0
            photometric_loss, smooth_loss, warped_images = multi_scale_flow_loss(
                prev_image_gt, next_image_gt,
                network_output, flow_mask)
            loss = photometric_loss + self.options.smooth_weight * smooth_loss
            losses['flow'] = loss
            losses['flow_photometric'] = photometric_loss
            losses['flow_smoothness'] = smooth_loss
            outputs.update( { 'warped': (warped_images[-1] + 1.) / 2.,
                              'flow_vis': flow2rgb(network_output[-1]) } )
        elif self.options.model == 'recons':
            loss = self.image_loss(network_output[-1], next_image_gt)
            losses['perceptual'] = loss
            outputs['recons_image'] = (network_output[-1] + 1.) / 2.
        loss.backward()
        optimizer.step()

        event_vis = torch.sum(event_volume, dim=1, keepdim=True)

        outputs.update({ 'raw_events': torch.clamp(event_vis, 0, 1), 
                         'gt_gray': (next_image + 1.) / 2. })
        
        return losses, outputs

    """
    Generate Tensorboard summaries.
    """
    def summaries(self, input, losses, output, mode="train"):
        nrow = 4 
        self.summary_writer.add_scalar("{}/learning_rate".format(mode),
                                       self.get_lr(),
                                       self.step_count)
        for k, v in losses.items():
            self.summary_writer.add_scalar("{}/{}".format(mode, k), v, self.step_count)
        for k, v in output.items():
            images = make_grid(v, nrow=nrow)
            self.summary_writer.add_image("{}/{}".format(mode, k),
                                          make_grid(v, nrow=nrow), self.step_count)

    def train_summaries(self, input_batch, losses, output):
        self.summaries(input, losses, output, mode="train")
