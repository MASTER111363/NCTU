# HAPPYNET(HN)

# Requirement
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision/)
* [tabulate](https://pypi.python.org/pypi/tabulate/)

# Usage
To run HN use the following command:

```bash
python3 testac.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH>  --model=PreResNet164 --epochs=225 --lr_init=0.1 --wd=3e-4 --swa --swa_start=126 --swa_lr=0.05
``` 
Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```PATH``` &mdash; path to the data directory

# Results

## CIFAR-100

Test accuracy (%) of SGD and SWA on CIFAR-100 for different training budgets. For each model the _Budget_ is defined as the number of epochs required to train the model with the conventional SGD procedure.

|                           | SWA 1.5 Budgets |
| ------------------------- |:---------------:|
| PreResNet164 (150)        | 80.35 ± 0.16    |
