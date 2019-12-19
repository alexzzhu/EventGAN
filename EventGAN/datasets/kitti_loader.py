import torch
from torch.utils.data import Dataset
import random
import numpy as np
import random
import os
import cv2

from IPython.core.debugger import Pdb
from utils.event_utils import gen_discretized_event_volume
import pytorch_utils
import torchvision
from PIL import Image

class KITTIObjectSequence(Dataset):
    def __init__(self, args, path, image_size, n_skip_frames=-1):
        super(KITTIObjectSequence, self).__init__()

        self.args = args
        self.path = path

        self.n_ima = self.count_n_pngs(os.path.join(path, "image_2"))
        
        # store for center crop
        self.top_left = self.args.top_left
        self.image_size = image_size
        self.max_skip_frames = 3 #self.args.max_skip_frames
        self.flip_x = self.args.flip_x
        self.flip_y = self.args.flip_y
        self.n_skip_frames = n_skip_frames

    def count_n_pngs(self, path):
        files = os.listdir(path)
        files = [f for f in files if '.png' in f]

        return len(files)
        
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = cv2.resize(image, self.image_size)
        image = cv2.GaussianBlur(image, (3, 3), 0)[..., np.newaxis]
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255.
        image = (image - 0.5) * 2.       
        
        return image
        
    def __len__(self):
        """ Return the first frame that has number_events before it """
        return self.n_ima

    def get_single_item(self, ind):
        if self.n_skip_frames > 0:
            skip = self.n_skip_frames
        else:
            skip = 1 + int((self.max_skip_frames-1)*np.random.rand())
        
        next_image_path = os.path.join(self.path, "image_2", "{:06d}.png".format(ind))
        prev_image_path = os.path.join(self.path, "prev_2",
                                       "{:06d}_{:02d}.png".format(ind, skip))
        
        next_image = self.load_image(next_image_path)
        prev_image = self.load_image(prev_image_path)

        # Event volume is t-y-x
        output = { "prev_image" : prev_image.copy(),
                   "next_image" : next_image.copy() }
        return output

    def __getitem__(self, ind_in):
        return self.get_single_item(ind_in)

def main():
    import configs
    from pytorch_utils import BaseOptions
    options = BaseOptions()
    options.parser = configs.get_args(options.parser)
    args = options.parse_args()
    sequence = KITTIObjectSequence(args, "/NAS/data/khantk/training/image_2")
    dataloader = torch.utils.data.DataLoader(sequence, batch_size=8, shuffle=True, num_workers=0)
    dataloader = sequence
    for i_batch, sample_batched in enumerate(dataloader):
        #print sequence[2]['next_image'].shape
        break
    #print("Here")

if __name__== "__main__":
    main()
