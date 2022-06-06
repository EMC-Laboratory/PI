# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

from utils import *
from train import *
from config import Config
from tensorboardX import SummaryWriter
import dqn_model
import torch
import argparse
import warnings
import os
import numpy as np

from dqn_agent import Agent

warnings.filterwarnings("ignore")  # filterwarnings help to specify whether warnings are to be ignored, displayed,
#                                   raise errors. Control warning flow basically


def main():
    folder_init(opt)  # Functions makes needed folders if they don't exist. In util. Place to save stuff
    pre_epoch = 0
    best_score = 1e-8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # if there's GPU device, else use CPU
    # specifies where we're doing these tensor calculations and what not

    # Initialize model chosen
    try:
        policy = dqn_model.QNetwork(opt, seed=0)
    except KeyError('==> Your model is not found.'):
        exit(0)
    else:
        print("==> Model initialized successfully.")

    if opt.LOAD_SAVED_MOD:
        policy, pre_epoch, best_score = policy.load(opt, map_location=device.type)
    policy.best_score = best_score
    policy = policy.to(device)

    # Instantiation of tensorboard and add net graph to it
    writer = SummaryWriter(opt.SUMMARY_PATH)
    dummy_input = wrap_np(np.zeros(opt.STATE_LEN), device)
    try:
        writer.add_graph(policy, dummy_input)
    except KeyError:
        writer.add_graph(policy.module, dummy_input)

    # Start training or testing
    if not opt.MASS_TESTING:
        policy = training(opt, writer, policy, pre_epoch=pre_epoch)
        testing(opt, policy)
    else:
        testing(opt, policy)


def str2bool(b):
    if b.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # Options
    # This sections adds 'arguments' or choices.
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-lsm', '--LOAD_SAVED_MOD', type=str2bool,
                        help='If you want to load saved model')
    parser.add_argument('-und', '--USE_NEW_DATA', type=str2bool,
                        help='If you want to use new data')
    parser.add_argument('-gi', '--GPU_INDEX', type=str,
                        help='Index of GPUs you want to use')
    # parser.parse_args is basically giving a response to an argument. If no response is given, argument is None
    args = parser.parse_args()  # ['flag', 'value'] Need a flag so you know which argument is being talked
    #                                        is being answered. And a value for your response for the argument.
    #                                        parse_args will also return the populated namespace, argument and the res
    print(args)
    opt = Config() # creates a class object called opt with parameters specified by class Config from config.
    for k, v in vars(args).items():
        if v is not None and hasattr(opt, k):  # hasattr(object, name) checks if object has attribute name where name
        #                                        is a string. Return true if yes, false if no.
            setattr(opt, k, v)                 # If something from the added arguments contradicts the preset values
#                                                in Config, change the attribute option to match
            print(k, v, getattr(opt, k))       # Seems to just be a checker to show it is working
    if args.GPU_INDEX:  # My conclusion is that None is treated as false
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_INDEX
    main()
