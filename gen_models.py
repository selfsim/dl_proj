# Grid search over selected hyperparameters
from pathlib import Path
import argparse
from train import trainer
from models.archs import nn_models as models

lrs = ["1e-6", "1e-5", "1e-4", "1e-3"]
optimizers = ["adam", "rmsprop"]
batch_sizes = [64, 128, 256]
model_dir = "./models/saved_models"

def main(args):
    for lr in lrs:
        for opt in optimizers:
            for batch_size in batch_sizes:

                args.__dict__['batch_size'] = batch_size
                args.__dict__['optimizer'] = opt
                args.__dict__['lr'] = float(lr)

                path_string = "{0}/{1}_{2}_{3}_{4}".format(model_dir, args.__dict__['model'], batch_size, opt, lr)
                args.__dict__['save_path'] = Path(path_string)

                tr = trainer(args)
                tr.train_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search hyper parameters for a certain model")
    parser.add_argument('--model', default="linear", metavar='str', choices=models,
                        help='The model to be trained (default linear)')
    parser.add_argument('--epochs', type=int, default=5, metavar='n', choices = range(0,1000),
                        help='The number of epochs to train for (default 5)')
    parser.add_argument('--use-cuda', action='store_true', default=True, 
                        help='Whether or not to leverage GPU (default True)')
    
    args = parser.parse_args()
    
    args.__dict__['task'] = 'new'

    main(args)