# takes as input a configuration file and trains the model found within it
# uses GPU if enabled



import argparse
import torch


def main(config):
    # load the data