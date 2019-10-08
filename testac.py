import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision
import models
import utils
import tabulate
import SWA
from micronet_challenge import counting

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')


args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
model_cfg = getattr(models, args.model)
ds = getattr(torchvision.datasets, args.dataset)
path = os.path.join(args.data_path, args.dataset.lower())
train_set = ds(path, train=True, download=True, transform=model_cfg.transform_train)
test_set = ds(path, train=False, download=True, transform=model_cfg.transform_test)
loaders = {
    'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'test': torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
}

num_classes = max(train_set.train_labels) + 1
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()
swa_n = 0
criterion = F.cross_entropy
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)

checkpoint = torch.load('/home/billy/PycharmProjects/swa/SWA/checkpoint-225.pt')
model.load_state_dict(checkpoint['swa_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])


model.eval()

test_res = utils.eval(loaders['test'], model, criterion)
print(test_res)


from torchsummary import summary
summary(model, (3, 32, 32))

def read_block(block, input_size, f_activation='swish'):
  """Reads the operations on a single EfficientNet block.

  Args:
    block: efficientnet_model.MBConvBlock,
    input_shape: int, square image assumed.
    f_activation: str or None, one of 'relu', 'swish', None.

  Returns:
    list, of operations.
  """
  ops = []
  # 1
  l_name = '_expand_conv'
  if hasattr(block, l_name):
    layer = getattr(block, l_name)
    layer_temp = counting.Conv2D(
        input_size, layer.kernel.shape.as_list(), layer.strides, layer.padding,
        True, f_activation)  # Use bias true since batch_norm
    ops.append((l_name, layer_temp))
  # 2
  l_name = '_depthwise_conv'
  layer = getattr(block, l_name)
  layer_temp = counting.DepthWiseConv2D(
      input_size, layer.weights[0].shape.as_list(), layer.strides,
      layer.padding, True, f_activation)  # Use bias true since batch_norm
  ops.append((l_name, layer_temp))
  # Input size might have changed.
  input_size = counting.get_conv_output_size(
      image_size=input_size, filter_size=layer_temp.kernel_shape[0],
      padding=layer_temp.padding, stride=layer_temp.strides[0])
  # 3
  if block._has_se:
    se_reduce = getattr(block, '_se_reduce')
    se_expand = getattr(block, '_se_expand')
    # Kernel has the input features in its second dimension.
    n_channels = se_reduce.kernel.shape.as_list()[2]
    ops.append(('_se_reduce_mean', counting.GlobalAvg(input_size, n_channels)))
    # input size is 1
    layer_temp = counting.Conv2D(
        1, se_reduce.kernel.shape.as_list(), se_reduce.strides,
        se_reduce.padding, True, f_activation)
    ops.append(('_se_reduce', layer_temp))
    layer_temp = counting.Conv2D(
        1, se_expand.kernel.shape.as_list(), se_expand.strides,
        se_expand.padding, True, 'sigmoid')
    ops.append(('_se_expand', layer_temp))
    ops.append(('_se_scale', counting.Scale(input_size, n_channels)))

  # 4
  l_name = '_project_conv'
  layer = getattr(block, l_name)
  layer_temp = counting.Conv2D(
      input_size, layer.kernel.shape.as_list(), layer.strides, layer.padding,
      True, None)  # Use bias true since batch_norm, no activation
  ops.append((l_name, layer_temp))

  if (block._block_args.id_skip
      and all(s == 1 for s in block._block_args.strides)
      and block._block_args.input_filters == block._block_args.output_filters):
    ops.append(('_skip_add', counting.Add(input_size, n_channels)))
  return ops, input_size


def read_model(model, input_shape, f_activation='swish'):
  """Reads the operations on a single EfficientNet block.

  Args:
    model: efficientnet_model.Model,
    input_shape: int, square image assumed.
    f_activation: str or None, one of 'relu', 'swish', None.

  Returns:
    list, of operations.
  """
  # Ensure that the input run through model
  _ = model(tf.ones(input_shape))
  input_size = input_shape[1]  # Assuming square
  ops = []
  # 1
  l_name = '_conv_stem'
  layer = getattr(model, l_name)
  layer_temp = counting.Conv2D(
      input_size, layer.weights[0].shape.as_list(), layer.strides,
      layer.padding, True, f_activation)  # Use bias true since batch_norm
  ops.append((l_name, layer_temp))
  # Input size might have changed.
  input_size = counting.get_conv_output_size(
      image_size=input_size, filter_size=layer_temp.kernel_shape[0],
      padding=layer_temp.padding, stride=layer_temp.strides[0])

  # Blocks
  for idx, block in enumerate(model._blocks):
    block_ops, input_size = read_block(block, input_size,
                                       f_activation=f_activation)
    ops.append(('block_%d' % idx, block_ops))

  # Head
  l_name = '_conv_head'
  layer = getattr(model, l_name)
  layer_temp = counting.Conv2D(
      input_size, layer.weights[0].shape.as_list(), layer.strides,
      layer.padding, True, f_activation)  # Use bias true since batch_norm
  n_channels_out = layer.weights[0].shape.as_list()[-1]
  ops.append((l_name, layer_temp))

  ops.append(('_avg_pooling', counting.GlobalAvg(input_size, n_channels_out)))

  l_name = '_fc'
  layer = getattr(model, l_name)
  ops.append(('_fc', counting.FullyConnected(
      layer.kernel.shape.as_list(), True, None)))
  return ops
