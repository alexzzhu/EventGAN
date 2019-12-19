from utils.viz_utils import gen_event_images
from pytorch_utils import BaseOptions
from models.eventgan_base import EventGANBase
import configs
import cv2
import numpy as np
import torch

# Read in images.
prev_image = cv2.imread('EventGAN/example_figs/007203_01.png')
next_image = cv2.imread('EventGAN/example_figs/007203.png')

prev_image = cv2.resize(prev_image, (861, 260))
next_image = cv2.resize(next_image, (861, 260))

prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

images = np.stack((prev_image, next_image)).astype(np.float32)
images = (images / 255. - 0.5) * 2.
images = torch.from_numpy(images).cuda()

# Build network.
options = BaseOptions()
options.parser = configs.get_args(options.parser)
args = options.parse_args()

EventGAN = EventGANBase(args)
event_volume = EventGAN.forward(images, is_train=False)

event_images = gen_event_images(event_volume[-1], 'gen')

event_image = event_images['gen_event_time_image'][0].cpu().numpy().sum(0)

event_image *= 255. / event_image.max()
event_image = event_image.astype(np.uint8)

cv2.imwrite('simulated_event.png', event_image)
