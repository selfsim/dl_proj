# python ./test.py --help
from torch import nn as nn
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
from models.archs import *
import argparse
import pathlib
import torch
from torchvision.transforms.transforms import ToTensor

models = nn_models

def main(args):
    tester(args)

class tester():

    def __init__(self, args):
        self.vars = vars(args)

        preprocess = transforms.Compose(
            [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0,0,0), (0.25, 0.25, 0.25)),
            transforms.RandomErasing(),
            ]
        )

        self.data = datasets.CIFAR100(
            root="./../data",
            train=False,
            download=True,
            transform= preprocess if self.vars['preprocess'] else ToTensor()
        )

        self.dataloader = self.get_data_loader()

        self.load_model()

        self.test_model()

    def load_model(self):
        checkpoint = torch.load(self.vars['load_path'])
        self.vars.update(checkpoint)

        self.vars["device"] = torch.device("cuda" if self.vars["use_cuda"] == True and torch.cuda.is_available() else "cpu")
        self.vars['_model'] = self.get_model()
        self.vars["_model"] = self.get_model().to(self.vars["device"])
        self.vars['_model'].load_state_dict(checkpoint['model_state_dict'])
        
    def test_model(self):
        model = self.vars["_model"]
        device = self.vars['device']
        size = len(self.dataloader.dataset)

        model.eval()

        correct = 0

        # inference
        with torch.no_grad():
            for x,y in self.dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= size
        percent_correct = 100*correct
        print(f" Test Error: \n Accuracy: {(percent_correct):>0.1f}%\n")

        return percent_correct

    def get_data_loader(self):
        test_dataloader = DataLoader(self.data, shuffle=True)

        return test_dataloader

    def get_model(self):
        val = self.vars['model']
        if(val == "linear"):
            # c * h * w, # classes
            return linearClassifier(
                3 * 32 * 32,
                100)

        elif (val == "conv"):
            return convNet(100)
        
        elif (val == "google"):
            return googleNet(100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model on CIFAR100 data")
    parser.add_argument('--load-path', type= pathlib.Path, required=True, metavar='path', 
                        help='The path from which to load a model')                  
    parser.add_argument('--model', default="linear", metavar='str', choices=models,
                        help='The model to be trained (default linear)')
    parser.add_argument('--use-cuda', default=True, action='store_true')
    parser.add_argument('--no-use-cuda', dest='use-cuda', action='store_false')          
                        
    args = parser.parse_args()

    main(args)