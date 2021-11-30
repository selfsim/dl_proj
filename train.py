# python ./train.py --help
from comet_ml import Experiment, ExistingExperiment, API
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.archs import *
from sklearn.model_selection import train_test_split
from math import inf
from models.archs import nn_models
import argparse
import pathlib
import torch
import torch.optim as optim

torch.manual_seed(1)

# logging API vars
api_key = "33KRTNcG2iva0PuiNfXhEqcRr"
api = API(api_key=api_key)

def main(args):

    # # # init training environment # # #

    tr = trainer(args)
    
    # # # train and save the model # # #

    tr.train_model()    

class trainer():    

    classes = 100

    experiment = None    

    preprocess = transforms.Compose(
        [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0,0,0), (0.25, 0.25, 0.25)),
        transforms.RandomErasing(),
        ]
    )

    cifar100_data = datasets.CIFAR100(
        root="./../data",
        train=True,
        download=True,
        transform=preprocess
    )

    cifar100_test_data = datasets.CIFAR100(
        root="./../data",
        train=False,
        download=True,
        transform=preprocess
    )


    def __init__(self, args):
        self.vars = vars(args)  

        if(self.vars['task'] == 'load'):
            self.load_model()
        else:
            self.init_model()

    # # # model initialization functions # # #
    def init_model(self):

        # logging purposes
        trainer.experiment = Experiment(
            api_key="33KRTNcG2iva0PuiNfXhEqcRr",
            project_name="dl",
            workspace="selfsim",
        )

        # initialize vars
        self.vars["device"] = torch.device("cuda" if self.vars["use_cuda"] == True and torch.cuda.is_available() else "cpu")
        self.vars['name'] = self.vars['save_path'].stem        
        self.vars['loss_function'] = "crossentropy"
        self.vars["current_epoch"] = 0;
        self.vars["loss"] = inf
        self.vars["_train_dataloader"], self.vars["_validation_dataloader"], self.vars["_test_dataloader"] = self.get_data_loaders()
        self.vars["_model"] = self.get_model().to(self.vars["device"])
        self.vars["_optimizer"] = self.get_opt()
        self.vars['_scheduler'] = self.get_scheduler()
        self.vars["_loss_function"] = self.get_loss_function()
        
    def load_model(self):

        # set up variable tracking 
        self.vars['name'] = self.vars['save_path'].stem
        api_experiment = api.get_experiment("selfsim", "dl", self.vars["name"])
        self.vars['experiment_key'] = api_experiment.key
        
        trainer.experiment = ExistingExperiment(
            api_key="33KRTNcG2iva0PuiNfXhEqcRr",
            previous_experiment= self.vars['experiment_key'],
            project_name="dl",
            workspace="selfsim",
        )

        load_path = self.vars['load_path']
        
        step = int(api_experiment.get_parameters_summary("curr_step")['valueCurrent'])
        epoch = int(api_experiment.get_parameters_summary("curr_epoch")['valueCurrent'])

        trainer.experiment.set_step(step)
        trainer.experiment.set_epoch(epoch)

        # load information from file
        checkpoint = torch.load(load_path)
        self.vars.update(checkpoint)

        # init vars
        self.vars["device"] = torch.device("cuda" if self.vars["use_cuda"] == True and torch.cuda.is_available() else "cpu")
        self.vars["_train_dataloader"], self.vars["_validation_dataloader"], self.vars["_test_dataloader"] = self.get_data_loaders()
        self.vars['_model'] = self.get_model()
        self.vars["_model"] = self.get_model().to(self.vars["device"])
        self.vars['_optimizer'] = self.get_opt()
        self.vars['_loss_function'] = self.get_loss_function()
        self.vars['_scheduler'] = self.get_scheduler()
        self.vars['_model'].load_state_dict(checkpoint['model_state_dict'])
        self.vars['_optimizer'].load_state_dict(checkpoint['optim_state_dict'])

    def save_model(self):
        torch.save({
            'current_epoch' : self.vars['current_epoch'],
            'model_state_dict': self.vars["_model"].state_dict(),
            'optim_state_dict': self.vars["_optimizer"].state_dict(),
            'loss' : self.vars['loss'],
            'loss_function': self.vars['loss_function'],
            'lr': self.vars['lr'],
            'optimizer' : self.vars['optimizer'],
            '_model': self.vars["_model"],
            '_optimizer' : self.vars["_optimizer"],
            'model': self.vars['model'],
            'experiment_key' : trainer.experiment.get_key()
        }, self.vars["save_path"])

    # # # model training functions
    def train_model(self):
        trainer.experiment.set_name(self.vars['name'])
        trainer.experiment.log_parameter("lr", self.vars['lr'])
        trainer.experiment.log_parameter("batch_size", self.vars['batch_size'])
        trainer.experiment.log_parameter("optimizer", self.vars['optimizer'])

        scheduler = self.vars["_scheduler"]
        validation_dataloader = self.vars["_validation_dataloader"]
        test_dataloader = self.vars["_test_dataloader"]

        current_epoch = self.vars["current_epoch"];
        epochs = self.vars['epochs']
        ceiling = current_epoch + epochs

        for e in range(current_epoch, ceiling):
            print(f"Epoch {e+1}\n---------------")

            train_loss, training_acc = self.train()

            trainer.experiment.log_metric("train_loss", train_loss, epoch=e)
            trainer.experiment.log_metric("training_acc", training_acc, epoch=e)

            val_loss, val_acc = self.test(validation_dataloader, "Validation")

            trainer.experiment.log_metric("val_loss", val_loss, epoch=e)
            trainer.experiment.log_metric("val_acc", val_acc, epoch=e)

            # for logging purposes
            test_loss, test_acc = self.test(test_dataloader, "Testing")
            trainer.experiment.log_metric("test_loss", test_loss, epoch=e)
            trainer.experiment.log_metric("test_acc", test_acc, epoch=e)

            scheduler.step(val_loss)

            self.save_model()

            self.vars["current_epoch"] += 1

    def train(self):
        dataloader = self.vars["_train_dataloader"]
        device = self.vars['device']
        model = self.vars["_model"]
        optimizer = self.vars["_optimizer"]
        loss_function = self.vars["_loss_function"]
        size = len(dataloader.dataset)

        model.train()

        loss, correct = 0, 0

        for batch, (x,y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # loss calculation    
            loss = loss_function(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # backprop
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  

        correct /= size
        percent_correct = 100*correct

        return loss, percent_correct   
    
    def test(self, dataloader, string):
        model = self.vars["_model"]
        device = self.vars['device']
        loss_function = self.vars["_loss_function"]
        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        model.eval()

        loss, correct = 0, 0

        # inference
        with torch.no_grad():
            for x,y in dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss += loss_function(pred,y).item()
                # 0/1 loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss /= num_batches
        correct /= size
        percent_correct = 100*correct
        print(string + f" Error: \n Accuracy: {(percent_correct):>0.1f}%, Avg loss: {loss:>8f} \n")

        return loss, percent_correct
    
    # # # configuration helper functions # # #
    def get_data_loaders(self):
        
        dataset = trainer.cifar100_data
        test_data = trainer.cifar100_test_data
        # Stratified split of 20% of training data into validation data
        # https://linuxtut.com/en/c6023453e00bfead9e9f/
        # This was allowed as mentioned in class

        #Split dataset into train and validation
        train_indices, val_indices = train_test_split(list(range(len(dataset.targets))), test_size=0.2, stratify=dataset.targets)
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        #Create DataLoader
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.vars['batch_size'], shuffle=True )
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.vars['batch_size'], shuffle=True)

        test_dataloader = DataLoader(test_data, batch_size=self.vars['batch_size'], shuffle=True)

        return train_data_loader, val_data_loader, test_dataloader

    def get_model(self):
        val = self.vars['model']
        if(val == "linear"):
            # c * h * w, # classes
            return linearClassifier(
                3 * 32 * 32,
                trainer.classes)

        elif (val == "conv"):
            return convNet(trainer.classes)
        
        elif (val == "google"):
            return googleNet(trainer.classes)

    def get_opt(self):
        val = self.vars['optimizer']
        if(val == "adam"):
            return optim.Adam(params=self.vars['_model'].parameters(), lr=self.vars['lr'])
        if(val == "rmsprop"):
            return optim.RMSprop(params=self.vars['_model'].parameters(), lr=self.vars['lr'])
        if(val == "adagrad"):
            return optim.Adagrad(params=self.vars['_model'].parameters(), lr=self.vars['lr'])

    def get_scheduler(self):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.vars['_optimizer'], 
            'min', patience=5, 
            threshold=1e-3
        )

    def get_loss_function(self):
        val = self.vars['loss_function']
        if(val == "crossentropy"):
            return nn.CrossEntropyLoss()        

# # # cli parse # # #            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR100 data")
    parser.add_argument('--task', default="new", metavar='str', choices=["new", "load"],
                        help='The task to be performed (train NEW model, LOAD from checkpoint)')
    parser.add_argument('--model', default="linear", metavar='str', choices=nn_models,
                        help='The model to be trained (default linear)')
    parser.add_argument('--epochs', type=int, default=5, metavar='n', choices = range(0,1000),
                        help='The number of epochs to train for (default 5)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='n', choices = range(1,5000),
                        help='The input training batch size (default 64)')               
    parser.add_argument('--optimizer', default="adam", metavar='str', choices=["adam", "rmsprop", "adagrad"],
                        help='Which optimizer to use (default adam)')
    parser.add_argument('--lr', default=0.001, type=float, metavar='f',
                        help='The initial learning rate (default 0.001)')    
    parser.add_argument('--load-path', type= pathlib.Path, required=False, metavar='path', 
                        help='The path from which to load a model')    
    parser.add_argument('--save-path', type= pathlib.Path, required=True, metavar='path',
                        help='The path at which to save the model')
    parser.add_argument('--use-cuda', default=True, action='store_true')
    parser.add_argument('--no-use-cuda', dest='use-cuda', action='store_false')      

    args = parser.parse_args()

    main(args)