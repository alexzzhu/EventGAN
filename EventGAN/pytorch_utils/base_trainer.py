from __future__ import division
import sys
import time

import torch
from tqdm import tqdm
tqdm.monitor_interval = 0
from tensorboardX import SummaryWriter

from pytorch_utils import CheckpointDataLoader, CheckpointSaver

class BaseTrainer(object):
    """ BaseTrainer class to be inherited

    options
    - time_to_run
    - checkpoint_dir
    - summary_dir
    - checkpoint
    - resume
    - num_epochs
    - batch_size
    - num_workers
    - pin_memory
    - shuffle_train
    - summary_steps
    - checkpoint_steps
    - test_steps

    init_fn needs to define:
    - models_dict
    - optimizers_dict
    - train_ds - training dataset to feed to the CheckpointDataLoader
    - cdl_kwargs - Optional - kwargs to feed to the CheckpointDataLoader
                   there are defaults available and can be overwritten
                   or just added to
    """

    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run \
            if self.options.time_to_run > 0 else -1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # keyword arguments for CheckpointDataLoader in the training
        # loop
        self.cdl_kwargs = {
                    "batch_size": self.options.batch_size,
                    "num_workers": self.options.num_workers,
                    "pin_memory": self.options.pin_memory,
                    "shuffle": self.options.shuffle_train
                }

        # override this function to define your model, optimizers etc.
        self.init_fn()

        self.models_dict = {k:v.to(self.device)
                for k,v in self.models_dict.items()}

        # tensorboardX SummaryWriter for use in train_summaries
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        # Load the latest checkpoints
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict,
                    self.optimizers_dict,
                    checkpoint_file=self.options.checkpoint)

        self.cdl_kwargs["checkpoint"] = self.checkpoint
            
        # Reload epoch and step count if a checkpoint was loaded
        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']
        self.epoch = self.epoch_count
        self.lr_schedulers = {k: torch.optim.lr_scheduler.ExponentialLR(v,
                                                gamma=self.options.lr_decay,
                                                last_epoch=self.epoch_count-1)\
                              for k,v in self.optimizers_dict.items()}

        #for opt in self.optimizers_dict:
        #    self.lr_schedulers[opt].step()

    def train(self):
        # Create the dataloader that will generate the data
        # permutation for each epoch
        train_data_loader = CheckpointDataLoader(self.train_ds,
                **self.cdl_kwargs)

        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs),
                total=self.options.num_epochs, initial=self.epoch_count):
            # setup the next epoch inside of train_data_loader
            # this will include the next dataset permutation
            train_data_loader.next_epoch(self.checkpoint)

            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                if self.endtime < 0 or time.time() < self.endtime:
                    batch = {k: v.to(self.device) for k,v in batch.items()}
                    out = self.train_step(batch)

                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(batch, *out)
                    
                    #if self.step_count % self.options.checkpoint_steps == 0:
                    #    self.saver.save_checkpoint(self.models_dict,
                    #            self.optimizers_dict, epoch, step+1,
                    #            self.options.batch_size,
                    #            train_data_loader.get_dataset_perm(),
                    #            self.step_count)
                    #
                    #    tqdm.write('Checkpoint saved')
                
                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                    self.step_count += 1
                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(self.models_dict,
                            self.optimizers_dict, epoch, step,
                            self.options.batch_size,
                            train_data_loader.get_dataset_perm(),
                            self.step_count) 
                    tqdm.write('Checkpoint saved')
                    return

            # apply the learning rate scheduling policy
            for opt in self.optimizers_dict:
                self.lr_schedulers[opt].step()
            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # Reset batch idx for the new epoch.
            train_data_loader.checkpoint_batch_idx = 0
            # save checkpoint after each epoch
            self.saver.save_checkpoint(self.models_dict,
                                       self.optimizers_dict, epoch+1, 0,
                                       self.options.batch_size, None, self.step_count)
            self.epoch = epoch
            
        self.test()
        if self.endtime > 0:
            # Throw an error to indicate completion of training if timed training is enabled.
            raise ValueError("Completed training, raising error to kill this job")
        return

    def get_lr(self):
        return next(iter(self.lr_schedulers.values())).get_lr()[0]

    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch, model_output):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self):
        pass
