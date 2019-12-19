import torch
from torch.utils.data import Dataset
import numpy as np
import random
import h5py
from utils.event_utils import gen_discretized_event_volume
import pytorch_utils

class MVSECSequence(Dataset):
    def __init__(self, args, train=True, path=None, start_time=-1):
        super(MVSECSequence, self).__init__()

        self.train = train
        self.args = args

        # store for center crop
        self.top_left = self.args.top_left
        self.image_size = self.args.image_size
        
        if start_time == -1:
            self.start_time = self.args.start_time
        else:
            self.start_time = start_time
            
        self.max_skip_frames = self.args.max_skip_frames

        self.flip_x = self.args.flip_x
        self.flip_y = self.args.flip_y

        if path is None:
            if self.train:
                self.path = args.train_file
            else:
                self.path = args.validation_file
                self.flip_x = 0
                self.flip_y = 0
        else:
            self.path = path

        # load and close to avoid subprocesses from referencing
        # the same file descriptor
        self.load(True)
        self.close()

    def load(self, only_length=False):
        self.sequence = h5py.File(self.path, 'r')

        if 'left' in self.sequence['davis']:
            davis_cam = self.sequence['davis']['left']
            # dense array of raw indexes NxHxW
            self.images = davis_cam['image_raw']
            
            # dense array of all events Nx4 (x,y,t,p)
            self.events = davis_cam['events']
            
            # list of ending event inds for each image Nx1
            self.image_to_event = davis_cam['image_raw_event_inds']
            
            # list of image times for each image Nx1
            self.images_ts = davis_cam['image_raw_ts']
        else:
            davis_cam = self.sequence['davis']
            self.images = davis_cam['image_raw']['data']
            self.events = davis_cam['events']['data']
            self.image_to_event = davis_cam['image_raw']['event_index']
            self.images_ts = davis_cam['image_raw']['timestamps']
            
        self.raw_image_size = self.images.shape[1:]
        self.start_frame = np.searchsorted(self.images_ts, self.start_time + self.images_ts[0])

        self.num_images = self.images.shape[0]
        self.loaded = True

    def close(self):
        self.images = None

        self.events = None
        self.image_to_event = None
            
        self.images_ts = None

        self.sequence.close()
        self.sequence = None
        self.loaded = False

    def __len__(self):
        """ Return the first frame that has number_events before it """
        length = self.num_images - self.start_frame - self.max_skip_frames - 1

        return length

    def get_prev_next_inds(self, ind):
        pind = self.start_frame+ind
        if self.train:
            cind = self.start_frame +ind + 1 + int((self.max_skip_frames-1)*np.random.rand())
        else:
            cind = pind + 2
        return pind, cind

    def get_box(self):
        top_left = self.top_left
        if self.train:
            top =  int(np.random.rand()*(self.raw_image_size[0]-1-self.image_size[0]))
            left = int(np.random.rand()*(self.raw_image_size[1]-1-self.image_size[1]))
            top_left = [top, left]
        bottom_right = [top_left[0]+self.image_size[0],
                        top_left[1]+self.image_size[1]]

        return top_left, bottom_right

    def get_image(self, ind, bbox):
        top_left, bottom_right = bbox
        image = self.images[ind][top_left[0]:bottom_right[0],
                                 top_left[1]:bottom_right[1],None]

        image = image.transpose((2,0,1)).astype(np.float32)/255.
        image -= 0.5
        image *= 2.

        image_ts = self.images_ts[ind]
        return image, image_ts

    def count_events(self, pind, cind):
        return self.image_to_event[cind] - self.image_to_event[pind]
        
    def get_events(self, pind, cind, bbox):
        top_left, bottom_right = bbox
        peind = max(self.image_to_event[pind], 0)
        ceind = self.image_to_event[cind]

        events = self.events[peind:ceind,:]
        mask = np.logical_and(np.logical_and(events[:,1]>=top_left[0],
                                             events[:,1]<bottom_right[0]),
                              np.logical_and(events[:,0]>=top_left[1],
                                             events[:,0]<bottom_right[1]))

        events_masked = events[mask]
        events_shifted = events_masked
        events_shifted[:,0] = events_masked[:, 0] - top_left[1]
        events_shifted[:,1] = events_masked[:, 1] - top_left[0]

        # subtract out min to get delta time instead of absolute
        events_shifted[:,2] -= np.min(events_shifted[:,2])

        # convolution expects 4xN
        #events_shifted = np.transpose(events_shifted).astype(np.float32)
        events_shifted = events_shifted.astype(np.float32)

        return events_shifted

    def get_num_events(self, pind, cind, bbox, dataset):

        peind = self.image_to_event[dataset][pind]
        ceind = self.image_to_event[dataset][cind]

        events = self.events[dataset][peind:ceind,:]
        return events.shape[0]
    
    def select_valid_input(self, ind_in):
        
        for i in reversed(range(len(self.cum_num_images))):
            if ind_in >= self.cum_num_images[i]:
                dataset = i
                ind = ind_in - self.cum_num_images[i]
                break
       
        pind, cind = self.get_prev_next_inds(ind, dataset)
        bbox = self.get_box(dataset)
        return self.get_num_events(pind, cind, bbox, dataset), \
                cind, pind, bbox, dataset

    def normalize_event_volume(self, event_volume):
        event_volume_flat = event_volume.view(-1)
        nonzero = torch.nonzero(event_volume_flat)
        nonzero_values = event_volume_flat[nonzero]
        if nonzero_values.shape[0]:
            lower = torch.kthvalue(nonzero_values,
                                   max(int(0.02 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            upper = torch.kthvalue(nonzero_values,
                                   max(int(0.98 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            max_val = max(abs(lower), upper)
            event_volume = torch.clamp(event_volume, -max_val, max_val)
            event_volume /= max_val
        return event_volume

    def apply_illum_augmentation(self, prev_image, next_image,
                                 gain_min=0.8, gain_max=1.2, gamma_min=0.8, gamma_max=1.2):
        random_gamma = gamma_min + random.random() * (gamma_max - gamma_min)
        random_gain = gain_min + random.random() * (gain_max - gain_min);
        prev_image = self.transform_gamma_gain_np(prev_image, random_gamma, random_gain)
        next_image = self.transform_gamma_gain_np(next_image, random_gamma, random_gain)
        return prev_image, next_image

    def transform_gamma_gain_np(self, image, gamma, gain):
        # apply gamma change and image gain.
        image = (1. + image) / 2.
        image = gain * np.power(image, gamma) 
        image = (image - 0.5) * 2.
        return np.clip(image, -1., 1.) 
         
    def get_single_item(self, ind):
        # this is so we know we there will be number_events
        # at the end of filtering
        if self.train:
            max_n_events = 100 #50000 / 6
            n_events = -1
            n_iters = 0
            while n_events < max_n_events:
                n_events = self.count_events(ind, ind+1)
                n_iters += 1
                if n_events < max_n_events:
                    ind = random.randint(0, self.__len__())
                
        pind, cind = self.get_prev_next_inds(ind)
        bbox = self.get_box()
        
        next_image, next_image_ts = self.get_image(cind, bbox)
        prev_image, prev_image_ts = self.get_image(pind, bbox)

        events = self.get_events(pind, cind, bbox)
        event_volume = gen_discretized_event_volume(torch.from_numpy(events).cpu(),
                                                    [self.args.n_time_bins*2,
                                                     self.image_size[0],
                                                     self.image_size[1]])
        if self.args.normalize_events:
            event_volume = self.normalize_event_volume(event_volume)

        prev_image_gt, next_image_gt = prev_image, next_image
        
        if self.train:
            if np.random.rand() < self.flip_x:
                event_volume = torch.flip(event_volume, dims=[2])
                prev_image = np.flip(prev_image, axis=2)
                next_image = np.flip(next_image, axis=2)
            if np.random.rand() < self.flip_y:
                event_volume = torch.flip(event_volume, dims=[1])
                prev_image = np.flip(prev_image, axis=1)
                next_image = np.flip(next_image, axis=1)
            prev_image_gt, next_image_gt = prev_image, next_image
            if self.args.appearance_augmentation:
                prev_image, next_image = self.apply_illum_augmentation(prev_image, next_image)

        # Event volume is t-y-x
        output = { "prev_image" : prev_image.copy(),
                   "prev_image_gt" : prev_image_gt.copy(),
                   "prev_image_ts" : prev_image_ts,
                   "next_image" : next_image.copy(),
                   "next_image_gt" : next_image_gt.copy(),
                   "next_image_ts" : next_image_ts,
                   "event_volume" : event_volume }
        return output
                
    def __getitem__(self, ind_in):
        if not self.loaded:
            self.load()

        return self.get_single_item(ind_in)

class WeightedRandomSampler(pytorch_utils.data_loader.CheckpointSampler):
    """
    Samples from a data_source with weighted probabilities for each element.
    Weights do not need to sum to 1. 
    Typical use case is when you have multiple datasets, the weights for each dataset are
    set to 1/len(ds). This ensures even sampling amongst datasets with different lengths.
    weights - tensor with numel=len(data_source)
    
    """
    def __init__(self, data_source, weights):
        super(WeightedRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.weights = weights

    def next_dataset_perm(self):
        return torch.multinomial(self.weights, len(self.data_source), replacement=True).tolist()
        
def get_and_concat_datasets(path_file, options, train=True):
    ds_list = []
    ds_len_list = []
    with open(path_file) as f:
        paths = f.read().splitlines()
    for path_start in paths:
        if not path_start:
            break
        path, start_time = path_start.split(' ')
        start_time = float(start_time)
        ds_list.append(MVSECSequence(options,
                                     train=train,
                                     path=path,
                                     start_time=start_time))
        weight = np.sqrt(len(ds_list[-1]))
        if "indoor" in path:
            weight *= 2
        ds_len_list += [weight] * len(ds_list[-1])
    weights = 1. / torch.Tensor(ds_len_list)
    ds = torch.utils.data.ConcatDataset(ds_list)
    sampler = WeightedRandomSampler(ds,
                                    weights)
    return ds, sampler
