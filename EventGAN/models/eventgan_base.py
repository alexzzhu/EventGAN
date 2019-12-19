from models.unet import UNet
import torch
from utils.utils import get_latest_checkpoint

class EventGANBase(object):
    def __init__(self, options):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = UNet(num_input_channels=2*options.n_image_channels,
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
        latest_checkpoint = get_latest_checkpoint(options.checkpoint_dir)
        checkpoint = torch.load(latest_checkpoint)
        self.generator.load_state_dict(checkpoint["gen"])
        self.generator.to(self.device)
        
    def forward(self, images, is_train=False):
        if len(images.shape) == 3:
            images = images[None, ...]
        assert len(images.shape) == 4 and images.shape[1] == 2, \
            "Input images must be either 2xHxW or Bx2xHxW."
        if not is_train:
            with torch.no_grad():
                self.generator.eval()
                event_volume = self.generator(images)
            self.generator.train()
        else:
            event_volume = self.generator(images)

        return event_volume
