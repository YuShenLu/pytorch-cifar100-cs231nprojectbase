"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet20():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

#
# class ResNetCIFAR(nn.Module):
#     """A residual neural network as originally designed for CIFAR-10."""
#
#     class Block(nn.Module):
#         """A ResNet block."""
#
#         def __init__(self, f_in: int, f_out: int, downsample: bool = False):
#             super(ResNetCIFAR.Block, self).__init__()
#
#             stride = 2 if downsample else 1
#             self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(f_out)
#             self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
#             self.bn2 = nn.BatchNorm2d(f_out)
#             self.relu = nn.ReLU(inplace=True)
#
#             # No parameters for shortcut connections.
#             if downsample or f_in != f_out:
#                 self.shortcut = nn.Sequential(
#                     nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
#                     nn.BatchNorm2d(f_out),
#                 )
#             else:
#                 self.shortcut = nn.Sequential()
#
#         def forward(self, x: torch.Tensor):
#             out = self.relu(self.bn1(self.conv1(x)))
#             out = self.bn2(self.conv2(out))
#             out += self.shortcut(x)
#             return self.relu(out)
#
#     def __init__(self, plan, initializers, outputs: int = 10):
#         super(ResNetCIFAR, self).__init__()
#         outputs = outputs or 10
#
#         self.num_classes = outputs
#
#         # Initial convolution.
#         current_filters = plan[0][0]
#         self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn = nn.BatchNorm2d(current_filters)
#         self.relu = nn.ReLU(inplace=True)
#
#         # The subsequent blocks of the ResNet.
#         blocks = []
#         for segment_index, (filters, num_blocks) in enumerate(plan):
#             for block_index in range(num_blocks):
#                 downsample = segment_index > 0 and block_index == 0
#                 blocks.append(ResNetCIFAR.Block(current_filters, filters, downsample))
#                 current_filters = filters
#
#         self.blocks = nn.Sequential(*blocks)
#
#         # Final fc layer. Size = number of filters in last segment.
#         self.fc = nn.Linear(plan[-1][0], outputs)
#         self.criterion = nn.CrossEntropyLoss()
#
#         # # this part of code is for layer initialization
#         # for initializer in initializers:
#         #     initializer = Initializer(initializer)
#         #     self.apply(initializer.get_initializer())
#
#     def forward(self, x: torch.Tensor):
#         batch_size = x.shape[0]
#         out = self.relu(self.bn(self.conv(x)))
#         out = self.blocks(out)
#         out = F.avg_pool2d(out, int(out.shape[3]))
#         out = out.view(batch_size, -1)
#         out = self.fc(out)
#         return out
#
#     @staticmethod
#     def is_valid_model_name(model_name: str):
#         valid_model_names = [f"resnet_{layers}" for layers in (20, 56)]
#         return (model_name in valid_model_names)
#
#     @staticmethod
#     def get_model_from_name(model_name: str, initializers, outputs: int = 10):
#         """The naming scheme for a ResNet is ``'resnet_D[_W]'``.
#         D is the model depth (e.g. ``'resnet_56'``)
#         """
#
#         if not ResNetCIFAR.is_valid_model_name(model_name):
#             raise ValueError('Invalid model name: {}'.format(model_name))
#
#         depth = int(model_name.split('_')[-1])  # for resnet56, depth 56, width 16
#         if len(model_name.split('_')) == 2:
#             width = 16
#         else:
#             width = int(model_name.split('_')[3])
#
#         if (depth - 2) % 3 != 0:
#             raise ValueError('Invalid ResNetCIFAR depth: {}'.format(depth))
#         num_blocks = (depth - 2) // 6
#
#         model_arch = {
#             56: [(width, num_blocks), (2 * width, num_blocks), (4 * width, num_blocks)],
#             20: [(width, num_blocks), (2 * width, num_blocks), (4 * width, num_blocks)],
#         }
#
#         return ResNetCIFAR(model_arch[depth], initializers, outputs)
#
#
# # adapted from https://raw.githubusercontent.com/matthias-wright/cifar10-resnet/master/model.py
# # under the MIT license
# class ResNet9(nn.Module):
#     """A 9-layer residual network, excluding BatchNorms and activation functions.
#     Based on the myrtle.ai `blog`_ and Deep Residual Learning for Image Recognition (`He et al, 2015`_).
#     Args:
#         num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``10``.
#     .. _blog: https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
#     .. _He et al, 2015: https://arxiv.org/abs/1512.03385
#     """
#
#     def __init__(self, num_classes: int = 10):
#         super().__init__()
#
#         self.body = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=64, momentum=0.9),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=128, momentum=0.9),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             BasicBlock(inplanes=128, planes=128, stride=1),
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=256, momentum=0.9),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=256, momentum=0.9),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             BasicBlock(inplanes=256, planes=256, stride=1),
#         )
#
#         self.fc = nn.Linear(in_features=256, out_features=num_classes, bias=True)
#
#     def forward(self, x):
#         out = self.body(x)
#         out = F.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out



