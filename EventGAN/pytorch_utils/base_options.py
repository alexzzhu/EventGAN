import os
import json
import argparse
from collections import namedtuple

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class BaseOptions(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=3600, help='Total time to run in seconds')
        gen.add_argument('--resume', dest='resume', type=str_to_bool, default=True, help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=0, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true', help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false', help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=100, help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=2, help='Batch size')
        train.add_argument('--test_batch_size', type=int, default=32, help='Batch size')
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')
        shuffle_test = train.add_mutually_exclusive_group()
        shuffle_test.add_argument('--shuffle_test', dest='shuffle_test', action='store_true', help='Shuffle testing data')
        shuffle_test.add_argument('--no_shuffle_test', dest='shuffle_test', action='store_false', help='Don\'t shuffle testing data')
        train.set_defaults(shuffle_train=True, shuffle_test=True)
        train.add_argument('--summary_steps', type=int, default=300, help='Summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=10000, help='Chekpoint saving frequency')
        train.add_argument('--test_steps', type=int, default=500, help='Testing frequency')

        optim = self.parser.add_argument_group('Optim')
        optim.add_argument("--lr_decay", type=float,
                default=0.99, help="Exponential decay rate")
        optim.add_argument("--wd", type=float,
                default=1e-5, help="Weight decay weight")

    def parse_args(self):
        self.args = self.parser.parse_args()
        log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
        json_path = os.path.join(log_dir, 'config.json')
        if os.path.exists(json_path):
            self.args.from_json = json_path
        
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)
            json_args = namedtuple("json_args", json_args.keys())(**json_args)
            return json_args
        else:
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self.save_dump()
            return self.args

    def save_dump(self):
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return
