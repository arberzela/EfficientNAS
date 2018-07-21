import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils import initialize_weights
from model.shakeshakeblock import shake_shake, generate_alpha_beta
from model.shakedrop import shake_drop, generate_alpha_beta_single

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
                                     int(out_channels / 2) if out_channels % 2 == 0 else int(out_channels / 2) + 1,
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
    def __init__(self, in_channels, out_channels, filter_size, stride, branch_index):
        super(ResidualBranch, self).__init__()

        self.residual_branch = nn.Sequential()

        self.residual_branch.add_module('Branch_{}:ReLU_1'.format(branch_index),
                                        nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_1'.format(branch_index),
                                        nn.Conv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=filter_size,
                                                  stride=stride,
                                                  padding=round(filter_size / 3),
                                                  bias=False))
        self.residual_branch.add_module('Branch_{}:BN_1'.format(branch_index),
                                        nn.BatchNorm2d(out_channels))
        self.residual_branch.add_module('Branch_{}:ReLU_2'.format(branch_index),
                                        nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_2'.format(branch_index),
                                        nn.Conv2d(out_channels,
                                                  out_channels,
                                                  kernel_size=filter_size,
                                                  stride=1,
                                                  padding=round(filter_size / 3),
                                                  bias=False))
        self.residual_branch.add_module('Branch_{}:BN_2'.format(branch_index),
                                        nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.residual_branch(x)


class BasicBlock(nn.Module):
    def __init__(self, n_input_plane, n_output_plane, filter_size, res_branches, stride, shake_config):
        super(BasicBlock, self).__init__()

        self.shake_config = shake_config
        self.branches = nn.ModuleList([ResidualBranch(n_input_plane, n_output_plane, filter_size, stride, branch + 1) for branch in range(res_branches)])

        # Skip connection
        self.skip = nn.Sequential()
        if n_input_plane != n_output_plane or stride != 1:
            self.skip.add_module('Skip_connection',
                                 SkipConnection(n_input_plane, n_output_plane, stride))
                                 

    def forward(self, x):
        out = sum([self.branches[i](x) for i in range(len(self.branches))])
            
        if len(self.branches) == 1:
            if self.config.apply_shakeDrop:
                alpha, beta = generate_alpha_beta_single(out.size(), self.shake_config if self.training else (False, False, False), x.is_cuda)
                out = shake_drop(out, alpha, beta, self.config.death_rate, self.training)
            else:
                out = self.branches[0](x)
        else:
            if self.config.apply_shakeShake:
                alpha, beta = generate_alpha_beta(len(self.branches), x.size(0), self.shake_config if self.training else (False, False, False), x.is_cuda)
                branches = [self.branches[i](x) for i in range(len(self.branches))]
                out = shake_shake(alpha, beta, *branches)
            else:
                out = sum([self.branches[i](x) for i in range(len(self.branches))])

        return out + self.skip(x)


class ResidualGroup(nn.Module):
    def __init__(self, block, n_input_plane, n_output_plane, n_blocks, filter_size, res_branches, stride, shake_config):
        super(ResidualGroup, self).__init__()
        self.group = nn.Sequential()
        self.n_blocks = n_blocks

        # The first residual block in each group is responsible for the input downsampling
        self.group.add_module('Block_1',
                              block(n_input_plane,
                                    n_output_plane,
                                    filter_size,
                                    res_branches,
                                    stride=stride,
                                    shake_config=shake_config))

        # The following residual block do not perform any downsampling (stride=1)
        for block_index in range(2, n_blocks + 1):
            block_name = 'Block_{}'.format(block_index)
            self.group.add_module(block_name,
                                  block(n_output_plane,
                                        n_output_plane,
                                        filter_size,
                                        res_branches,
                                        stride=1,
                                        shake_config=shake_config))

    def forward(self, x):
        return self.group(x)


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        nn.Module.config = config
        self.nr_main_blocks = config.nr_main_blocks
        self.nr_residual_blocks = {'Group_1': config.nr_residual_blocks_1,
                                   'Group_2': config.nr_residual_blocks_2,
                                   'Group_3': config.nr_residual_blocks_3,
                                   'Group_4': config.nr_residual_blocks_4,
                                   'Group_5': config.nr_residual_blocks_5,
                                   #'Group_6': config.nr_residual_blocks_6,
                                   }
 
        self.widen_factors = {'Group_1': config.widen_factor_1,
                              'Group_2': config.widen_factor_2,
                              'Group_3': config.widen_factor_3,
                              'Group_4': config.widen_factor_4,
                              'Group_5': config.widen_factor_5,
                              #'Group_6': config.widen_factor_6,
                              }

        self.res_branches = {'Group_1': config.res_branches_1,
                             'Group_2': config.res_branches_2,
                             'Group_3': config.res_branches_3,
                             'Group_4': config.res_branches_4,
                             'Group_5': config.res_branches_5,
                             #'Group_6': config.res_branches_6,
                             }

        self.filters_size = {'Group_1': config.filters_size_1,
                             'Group_2': config.filters_size_2,
                             'Group_3': config.filters_size_3,
                             'Group_4': config.filters_size_4,
                             'Group_5': config.filters_size_5,
                             #'Group_6': config.filters_size_6,
                             }
        
        shake_config = (config.forward_shake, config.backward_shake,
                             config.shake_image)

        ##########
        self.model = nn.Sequential()

        if config.dataset == 'cifar10':
            depth = sum([config.nr_convs * self.nr_residual_blocks['Group_{}'.format(i)] + 2 for i in range(1, self.nr_main_blocks + 1)])
            print(' | Multi-branch ResNet-' + str(depth) + ' CIFAR-10')

            block = BasicBlock

            self.model.add_module('Conv_0',
                                  nn.Conv2d(3,
                                            config.initial_filters,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False))
            self.model.add_module('BN_0',
                                  nn.BatchNorm2d(config.initial_filters))

            feature_maps_in = int(round(config.initial_filters * self.widen_factors['Group_1']))
            self.model.add_module('Group_1',
                                  ResidualGroup(block, 
                                                config.initial_filters, 
                                                feature_maps_in, 
                                                self.nr_residual_blocks['Group_1'], 
                                                self.filters_size['Group_1'],
                                                self.res_branches['Group_1'],
                                                1, 
                                                shake_config))

            for main_block_nr in range(2, self.nr_main_blocks + 1):
                feature_maps_out = int(round(feature_maps_in * self.widen_factors['Group_{}'.format(main_block_nr)]))
                self.model.add_module('Group_{}'.format(main_block_nr),
                                      ResidualGroup(block, 
                                                    feature_maps_in, 
                                                    feature_maps_out, 
                                                    self.nr_residual_blocks['Group_{}'.format(main_block_nr)],
                                                    self.filters_size['Group_{}'.format(main_block_nr)],
                                                    self.res_branches['Group_{}'.format(main_block_nr)],
                                                    2 if main_block_nr in (self.nr_main_blocks, self.nr_main_blocks - 1) else 1, 
                                                    shake_config))
                feature_maps_in = feature_maps_out

            self.feature_maps_out = feature_maps_out
            self.model.add_module('ReLU_0',
                                  nn.ReLU(inplace=True))
            self.model.add_module('AveragePool',
                                  nn.AvgPool2d(8, stride=1))
            self.fc = nn.Linear(feature_maps_out, 10)

        self.apply(initialize_weights)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.feature_maps_out)
        x = self.fc(x)
        return x
