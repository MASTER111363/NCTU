# HAPPYNET

# Requirement
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision/)
* [tabulate](https://pypi.python.org/pypi/tabulate/)
# Usage
To run SWA use the following command:

```bash
python3 testac.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH>  --model=PreResNet164 --epochs=225 --lr_init=0.1 --wd=3e-4 --swa --swa_start=126 --swa_lr=0.05
``` 
Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```PATH``` &mdash; path to the data directory
