from pytorch_utils import BaseOptions
import configs

options = BaseOptions()
options.parser = configs.get_args(options.parser)
args = options.parse_args()

if args.model.lower() == 'eventgan':
    from models.eventgan_trainer import EventGANTrainer
    trainer = EventGANTrainer(args)
    trainer.train()
elif args.model.lower() == 'flow' or args.model.lower() == 'recons':
    import os
    import torch
    from models.flow_reconstruct_trainer import FlowReconstructTrainer
    trainer = FlowReconstructTrainer(args) 
    trainer.train()

    checkpoint_filename = os.path.abspath(os.path.join(args.checkpoint_dir,
                                                       '{}.pickle'.format(args.model.lower())))
    torch.save(trainer.cycle_unet, checkpoint_filename)
else:
    raise ValueError("Model {} not supported, please select from {EventGAN, flow, recons}"
                     .format(args.model))

