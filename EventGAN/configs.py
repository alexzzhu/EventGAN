import argparse

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args(parser):
    parser.add_argument("--normalize_events",
                        help='Normalize the event volume if true',
                        type=str_to_bool, default=True)
    parser.add_argument("--model", type=str,
                        default='EventGAN',
                        help="Model to train, choose from {EventGAN, flow, recons}.")
    parser.add_argument("--lrd", type=float,
                        help="Discriminator learning rate.",
                        default=5e-5)
    parser.add_argument("--lrg", type=float,
                        help="Generator learning rate.",
                        default=2e-4)
    parser.add_argument("--lrc", type=float,
                        help="Flow/recons learning rate.",
                        default=2e-4)
    parser.add_argument("--no_train_gan",
                        help="If true, GAN will not be trained.",
                        action="store_true")
    gan_group = parser.add_argument_group("GAN")
    gan_group.add_argument("--disc_iter", type=int,
                           help="No. of iterations discriminator is trained every training step.",
                           default=2)
    gan_group.add_argument("--gen_iter", type=int,
                           help="No. of iterations generator is trained every training step.",
                           default=1)
    gan_group.add_argument("--num_filter_disc", type=int,
                           help="No. of filters in the first layers of the disriminator.",
                           default=64)
    gan_group.add_argument("--cycle_recons",
                           help='Whether to use the reconstruction loss in the cycle branch.',
                           action='store_true')
    gan_group.add_argument("--cycle_recons_model", type=str,
                           help='Pretrained reconstruction model used in the cycle branch.',
                           default="e2i_prev")
    gan_group.add_argument("--cycle_recons_weight", type=float,
                           help="Weight of cycle recons loss.",
                           default=1.0)
    gan_group.add_argument("--cycle_flow",
                           help='Whether to use the flow loss in the cycle consistency branch.',
                           action='store_true')
    gan_group.add_argument("--cycle_flow_model", type=str,
                           help='Name of the pretrained flow model used in the cycle branch.',
                           default="e2i_prev")
    gan_group.add_argument("--cycle_flow_weight", type=float,
                           help="Weight of cycle flow loss.",
                           default=1.0)
    gan_group.add_argument("--sn",
                           help='Whether to use spectral norm.',
                           type=str_to_bool, default=True)
    gan_group.add_argument("--flip_label", type=float,
                           help="Probability of flipping the labels when training the GAN.",
                           default=0.1)
    gan_group.add_argument("--smooth_weight", type=float,
                           help="weight of smoothness loss in flow loss.",
                           default=0.5)
    gan_group.add_argument("--no_flow_smoothness",
                           help="If true, no smoothness loss is applied to flow in auxiliary.",
                           action="store_true")
    argument_group = parser.add_argument_group("Data loader")
    argument_group.add_argument("--train_file", type=str,
                                help="File with list of hdf5 files for training.",
                                default="EventGAN/data/comb_train_files.txt")
    argument_group.add_argument("--validation_file", type=str,
                                help="File with list of hdf5 files for validation.",
                                default="EventGAN/data/validation_files.txt")
    argument_group.add_argument("--image_size", type=int,
                                help="Final image size for predictions (HxW).",
                                default=[256, 320], nargs=2)
    argument_group.add_argument("--top_left", type=int,
                                help="Top left corner of crop - (d1,d2).",
                                default=[2, 13], nargs=2)
    argument_group.add_argument("--start_time", type=float,
                                help="Time to start reading from the dataset (s).",
                                default=45.)
    argument_group.add_argument("--max_skip_frames", type=int,
                                help="Maximum number frames to skip.",
                                default=6)
    argument_group.add_argument("--n_time_bins", type=int,
                                help="Number of bins along the time dimension.",
                                default=9)
    argument_group.add_argument("--n_image_channels", type=int,
                                help="Number of channels in the image.",
                                default=1)
    argument_group.add_argument("--flip_x", type=float,
                                help="Probability of flipping the volume in x.",
                                default=0.5)
    argument_group.add_argument("--flip_y", type=float,
                                help="Probability of flipping the volume in y.",
                                default=0.5)
    argument_group.add_argument("--appearance_augmentation",
                                type=str_to_bool, default=True,
                                help="Augment images with gamma and image gain.")
    
    return parser
