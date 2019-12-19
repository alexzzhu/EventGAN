import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_folder",
                    help="Folder with hdf5 files",
                    default="/NAS/data/cwang/dataset_aug15_h5py")
parser.add_argument("--output_name",
                    help="Name of output txt file",
                    default="new_train_files")
args = parser.parse_args()

files = os.listdir(args.data_folder)

with open('{}.txt'.format(args.output_name), 'w') as output_file:
    for rel_path in files:
        abs_path = os.path.join(args.data_folder, rel_path)
        output_file.write("{} 0.0\n".format(abs_path))
