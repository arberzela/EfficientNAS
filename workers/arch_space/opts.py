import argparse
import os
from torch.cuda import is_available as is_cuda
from collections import OrderedDict

class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser('PyTorch ResNet Training script')

        # general options
        parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
        parser.add_argument('--manual_seed', type=int, default=0, help='Manually set RNG seed')
        parser.add_argument('--gen', type=str, default='./gen', help='Path to save generated files')
        parser.add_argument('--num_threads', type=int, default=2, help='number of data loading threads')
        parser.add_argument('--irun', type=int, default=1, help='Run index: 1 | 2 | 3 | ...')

        # training options
        #parser.add_argument('--n_epochs', type=int, default=1800, help='Number of total epochs to run')
        parser.add_argument('--budget', type=float, default=30.0, help='Number of total seconds to run')
        parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size (1 = pure stochastic)')
        parser.add_argument('--test_only', action='store_true', default=False, help='Run on test set only')
        parser.add_argument('--valid_frac', type=float, default=0.1, help='fraction of training set used for validation')

        # checkpointing options
        parser.add_argument('--save', type=str, default='checkpoints', help='Directory in which to save checkpoints')
        parser.add_argument('--resume', type=str, default=None, help='Resume from the latest checkpoint in this directory')

        # optimization options
        parser.add_argument('--LR', type=float, default=0.1, help='initial learning rate')
        parser.add_argument('--lr_shape', type=str, default='cosine', choices=['multistep', 'cosine'])
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--nesterov', type=bool, default=True, help='use nesterov momentum with SGD')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
        parser.add_argument('--T_e', type=float, default=1.0, help='maximum number of iterations before the lr decays.')
        parser.add_argument('--T_mul', type=float, default=1.0, help='multiplicative factor.')

        # model options
        parser.add_argument('--arch', type=str, default='macro_model', choices=['shakeshake', 'resnet', 'wrn', 'resnext', 'pyramidnet', 'macro_model'], 
                            help='use macro_model for doing architecture search.')
        #parser.add_argument('--shortcut_type', type=str, default=None, choices=['A', 'B', 'C'])
        #parser.add_argument('--depth', type=int, default=26, help='ResNet depth: 18 | 34 | 50 | 101 | ...')                           # not used for macro_model
        #parser.add_argument('--base_width', type=int, default=32, help='Number of filters of the first block')                        # not used for macro_model
        #parser.add_argument('--bottleneck', type=bool, default=False, help='to use basicblock for CIFAR datasets (default: False)')   # not used for macro_model
        parser.add_argument('--n_classes', type=int, default=None, choices=[10, 100, 1000])
        parser.add_argument('--forward_shake', action='store_true', default=False, help='Sample random numbers during the forward pass')
        parser.add_argument('--backward_shake', action='store_true', default=False, help='Sample random numbers during the backward pass')
        parser.add_argument('--shake_image', action='store_true', default=False, help='Use a different random number for each image in the mini-batch')

        # Shake-Shake options
        parser.add_argument('--apply_shakeDrop', action='store_true', default=False, help='Sample random numbers during the backward pass')
        parser.add_argument('--apply_shakeShake', action='store_true', default=False, help='Use a different random number for each image in the mini-batch')

        # cutout
        parser.add_argument('--cutout', action='store_true', default=False, help='Apply cutout')
        parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
        parser.add_argument('--length', type=int, default=16, help='length of holes')

        # mixup
        parser.add_argument('--alpha', type=float, default=0.2, help='interpolation strength (uniform=1., ERM=0.)')

        # autoAugment options
        parser.add_argument('--auto_aug', action='store_true', default=False, help='Apply AutoAugment')

        # PyramidNet options
        #parser.add_argument('--pynet_alpha', type=int, default=200, help='number of new channel increases per depth (default: 200)')

        # ShakeDrop options
        parser.add_argument('--death_rate', type=float, default=0.5, help='ShakeDrop success probability')

        # Hyperparameters controlling network depth
        parser.add_argument('--nr_main_blocks', type=int, default=3, choices=[3, 4, 5], help='Network main blocks: 3 | 4 | 5')
        parser.add_argument('--nr_convs', type=int, default=2, choices=[1, 2, 3, 4], help='Number of convolutional layers in each residual branch')

        parser.add_argument('--nr_residual_blocks_1', type=int, default=4, help='Number of residual blocks in main block 1')
        parser.add_argument('--nr_residual_blocks_2', type=int, default=4, help='Number of residual blocks in main block 2')
        parser.add_argument('--nr_residual_blocks_3', type=int, default=4, help='Number of residual blocks in main block 3')
        parser.add_argument('--nr_residual_blocks_4', type=int, default=0, help='Number of residual blocks in main block 4')
        parser.add_argument('--nr_residual_blocks_5', type=int, default=0, help='Number of residual blocks in main block 5')

        parser.add_argument('--initial_filters', type=int, default=16, help='Number of filters of the first convolution')

        parser.add_argument('--widen_factor_1', type=float, default=2.0, help='Number of filters of the first block')
        parser.add_argument('--widen_factor_2', type=float, default=2.0, help='Number of filters of the second block')
        parser.add_argument('--widen_factor_3', type=float, default=2.0, help='Number of filters of the third block')
        parser.add_argument('--widen_factor_4', type=float, default=0.0, help='Number of filters of the fourth block')
        parser.add_argument('--widen_factor_5', type=float, default=0.0, help='Number of filters of the fifth block')

        parser.add_argument('--res_branches_1', type=int, default=2, help='Number of residual branches in the first main block')
        parser.add_argument('--res_branches_2', type=int, default=2, help='Number of residual branches in the second main block')
        parser.add_argument('--res_branches_3', type=int, default=2, help='Number of residual branches in the third main block')
        parser.add_argument('--res_branches_4', type=int, default=0, help='Number of residual branches in the fourth main block')
        parser.add_argument('--res_branches_5', type=int, default=0, help='Number of residual branches in the fifth main block')

        parser.add_argument('--filters_size_1', type=int, default=3, choices=[1, 3, 5], help='Filters size of the first block') 
        parser.add_argument('--filters_size_2', type=int, default=3, choices=[1, 3, 5], help='Filters size of the second block') 
        parser.add_argument('--filters_size_3', type=int, default=3, choices=[1, 3, 5], help='Filters size of the third block') 
        parser.add_argument('--filters_size_4', type=int, default=3, choices=[1, 3, 5], help='Filters size of the fourth block') 
        parser.add_argument('--filters_size_5', type=int, default=3, choices=[1, 3, 5], help='Filters size of the fifth block') 

        self._args = parser.parse_args()

        if not is_cuda():
            raise Exception('No CUDA installation found!')

        if not os.path.isdir(self._args.save):
            print('Creating directory where to save checkpoints...\n')
            os.makedirs(self._args.save)

        #if self._args.shortcut_type == None:
        #    self._args.shortcut_type == 'A'

        if self._args.dataset == 'cifar10' or self._args.dataset == 'svhn':
            self._args.n_classes = 10
        elif self._args.dataset == 'cifar100':
            self._args.n_classes = 100

        # change T_e accordingly if you specify min and max budgets different from 400 and 10800 respectively
        if self._args.budget == 3600:
            self._args.T_e = 514.2857
        elif self._args.budget == 10800:
            self._args.T_e = 720.0

    @property
    def config(self):
        return self._args

    @property
    def model_config(self):
        return OrderedDict([
            ('arch', self._args.arch),
            #('depth', self._args.depth),
            #('shortcut_type', self._args.shortcut_type),
            #('bottleneck', self._args.bottleneck),
            #('base_width', self._args.base_width),
            ('forward_shake', self._args.forward_shake),
            ('backward_shake', self._args.backward_shake),
            ('shake_image', self._args.shake_image),
            ('input_shape', (1, 3, 32, 32)),
            ('n_classes', self._args.n_classes),
        ])

    @property
    def optim_config(self):
        return OrderedDir([
            ('batch_size', self._args.batch_size),
            ('LR', self._args.LR),
            ('weight_decay', self._args.weight_decay),
            ('momentum', self._args.momentum),
            ('nesterov', self._args.nesterov),
            ('lr_shape', self._args.lr_shape),
            ('T_e', self._args.T_e),
            ('T_mul', self._args.T_mul),
        ])
