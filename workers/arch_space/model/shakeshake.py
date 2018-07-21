import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils import initialize_weights
from model.shakeshakeblock import shake_shake, generate_alpha_beta


class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SkipConnection, self).__init__()

        self.s1 = nn.Sequential()
        self.s1.add_module('Skip_1_AvgPool',
                           nn.AvgPool2d(1, stride=stride))
        self.s1.add_module('Skip_1_Conv',
                           nn.Conv2d(in_channels,
                                     int(out_channels / 2),
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False))

        self.s2 = nn.Sequential()
        self.s2.add_module('Skip_2_AvgPool',
                           nn.AvgPool2d(1, stride=stride))
        self.s2.add_module('Skip_2_Conv',
                           nn.Conv2d(in_channels,
                                     int(out_channels / 2),
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False))

        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = F.relu(x, inplace=False)
        out1 = self.s1(out1)

        out2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        out2 = self.s2(out2)

        out = torch.cat([out1, out2], dim=1)
        out = self.batch_norm(out)

        return out


class ResidualBranch(nn.Module):
    def __init__(self, in_channels, out_channels, stride, branch_index):
        super(ResidualBranch, self).__init__()

        self.residual_branch = nn.Sequential()

        self.residual_branch.add_module('Branch_{}:ReLU_1'.format(branch_index),
                                        nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_1'.format(branch_index),
                                        nn.Conv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=3,
                                                  stride=stride,
                                                  padding=1,
                                                  bias=False))
        self.residual_branch.add_module('Branch_{}:BN_1'.format(branch_index),
                                        nn.BatchNorm2d(out_channels))
        self.residual_branch.add_module('Branch_{}:ReLU_2'.format(branch_index),
                                        nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_2'.format(branch_index),
                                        nn.Conv2d(out_channels,
                                                  out_channels,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1,
                                                  bias=False))
        self.residual_branch.add_module('Branch_{}:BN_2'.format(branch_index),
                                        nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.residual_branch(x)


class BasicBlock(nn.Module):
    def __init__(self, n_input_plane, n_output_plane, stride, shake_config):
        super(BasicBlock, self).__init__()

        self.shake_config = shake_config
        
        self.residual_branch1 = ResidualBranch(n_input_plane, n_output_plane, stride, 1)
        self.residual_branch2 = ResidualBranch(n_input_plane, n_output_plane, stride, 2)

        # Skip connection
        self.skip = nn.Sequential()
        if n_input_plane != n_output_plane:
            self.skip.add_module('Skip_connection',
                                 SkipConnection(n_input_plane, n_output_plane, stride))

    def forward(self, x):
        x1 = self.residual_branch1(x)
        x2 = self.residual_branch2(x)

        alpha, beta = generate_alpha_beta(x.size(0), self.shake_config if self.training else (False, False, False), x.is_cuda)
        out = shake_shake(x1, x2, alpha, beta)

        return out + self.skip(x)


class ResidualGroup(nn.Module):
    def __init__(self, block, n_input_plane, n_output_plane, n_blocks, stride, shake_config):
        super(ResidualGroup, self).__init__()
        self.group = nn.Sequential()

        # The first residual block in each group is responsible for the input downsampling
        self.group.add_module('Block_1',
                              block(n_input_plane,
                                    n_output_plane,
                                    stride=stride,
                                    shake_config=shake_config))

        # The following residual block do not perform any downsampling (stride=1)
        for block_index in range(1, n_blocks):
            block_name = 'block{}'.format(block_index + 1)
            self.group.add_module(block_name,
                                  block(n_output_plane,
                                        n_output_plane,
                                        stride=1,
                                        shake_config=shake_config))

    def forward(self, x):
        return self.group(x)


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        self.k = config.base_width
        depth = config.depth
        shake_config = (config.forward_shake, config.backward_shake,
                             config.shake_image)

        ##########
        self.model = nn.Sequential()

        if config.dataset == 'cifar10':
            assert((depth - 2) % 6 == 0)
            n_blocks_per_group = (depth - 2) // 6
            print(' | ResNet-' + str(depth) + ' CIFAR-10')

            block = BasicBlock

            self.model.add_module('Conv_0',
                                  nn.Conv2d(3,
                                            16,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False))
            self.model.add_module('BN_0',
                                  nn.BatchNorm2d(16))
            self.model.add_module('Group_1',
                                  ResidualGroup(block,       16,   self.k, n_blocks_per_group, 1, shake_config))
            self.model.add_module('Group_2',
                                  ResidualGroup(block,   self.k, 2*self.k, n_blocks_per_group, 2, shake_config))
            self.model.add_module('Group_3',
                                  ResidualGroup(block, 2*self.k, 4*self.k, n_blocks_per_group, 2, shake_config))
            self.model.add_module('ReLU_0',
                                  nn.ReLU(inplace=True))
            self.model.add_module('AveragePool',
                                  nn.AvgPool2d(8, stride=1))
            self.fc = nn.Linear(4*self.k, 10)

        self.apply(initialize_weights)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 4*self.k)
        x = self.fc(x)
        return x
