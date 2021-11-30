Welcome to Jonas' CIFAR100 classifacation repo.

Features: 
Device agnostic training/testing platform for CIFAR100

Capable of saving models

Capable of loading models for training continuation or inference

Support for hyperparameter grid search

Exports data to Comet.ML for analysis

# # installation

pip3 install comet_ml

install pytorch

## Training:

From the command line:
python ./train.py --help

Example:
python ./train.py --model "linear" --save-path "./" --epochs 5

select from linear, conv, google models

## Testing:
python ./test.py --help
python ./test.py --load-path "./path/to/saved/model"

shows test accuracy


Some experimental results shown in results subdirectory

All code contained in this project is my own or from the pytorch intro tutorial, unless specified otherwise.
