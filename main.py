import tensorflow as tf
import numpy as np
import time, os, argparse
from model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--pre_train_epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='../CelebA')
    parser.add_argument('--data_path', type=str, default='./data_sets')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--is_training', type=str2bool, default=1)
    parser.add_argument('--training_size', type=int, default=202599)
    parser.add_argument('--input_size', type=int, default=108)
    parser.add_argument('--num_examples_per_epoch', type=int, default=80)
    parser.add_argument('--target_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--final_dim', type=int, default=64)
    parser.add_argument('--showing_height', type=int, default=8)
    parser.add_argument('--showing_width', type=int, default=8)
    parser.add_argument('--sample_dir', type=str, default='./samples')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--log_dir', type=str, default='./tensorboard_log')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        magan = MAGAN(args, sess)
        if args.is_training:
            print('Training starts')
            magan.train()
        else:
            print('Test')
            magan.generator_test()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n' '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Not expected boolean type')

if __name__ == "__main__":
    main()
