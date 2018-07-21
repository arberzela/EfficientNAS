import time
import os, sys
import argparse

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from .helper import configuration_cifar10

from hpbandster.core.worker import Worker

class CIFAR10_base(Worker):
	def __init__(self, eta, min_budget, max_budget, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.mainsourcepath = '.'
		self.path = os.path.join(self.torch_mainsourcepath, 'arch_space')
		self.eta = eta
		self.min_budget = min_budget
		self.max_budget = max_budget

	def compute(self, config, budget, config_id, working_directory):
		return(configuration_cifar10(config=CIFAR10_base.complete_config(config), 
		                             budget=int(budget), 
		                             min_budget=self.min_budget,
		                             eta=self.eta,
		                             config_id=config_id,
		                             directory=working_directory, 
		                             source=self.path))
		
	@staticmethod
	def complete_config(config):
		config['nr_main_blocks'] = 3
		config['nr_convs'] = 2
		return(config)

	@staticmethod
	def get_config_space():
	    config_space=CS.ConfigurationSpace()

	    # architecture hyperparameters
	    config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('nr_residual_blocks_1', lower=1, upper=16, log=True))
	    config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('nr_residual_blocks_2', lower=1, upper=16, log=True))
	    config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('nr_residual_blocks_3', lower=1, upper=16, log=True))
	    config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('initial_filters', lower=8, upper=32, log=True))
	    config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('widen_factor_1', lower=0.5, upper=8, log=True))
	    config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('widen_factor_2', lower=0.5, upper=4, log=True))
	    config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('widen_factor_3', lower=0.5, upper=4, log=True))
	    config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('res_branches_1', lower=1, upper=5, log=False))
	    config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('res_branches_2', lower=1, upper=5, log=False))
	    config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('res_branches_3', lower=1, upper=5, log=False))
	    # other hyperparameters
	    config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('learning_rate', lower=1e-3, upper=1, log=True))
	    config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('batch_size', lower=32, upper=128, log=True))
	    config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-3, log=True))
	    config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('momentum', lower=1e-3, upper=0.99, log=False))
	    config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('alpha', lower=0, upper=1, log=False))
	    config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('length', lower=0, upper=20, log=False))
	    config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('death_rate', lower=0, upper=1, log=False))

	    return(config_space)

	@classmethod
	def data_subdir(cls):
		return('CIFAR10')

