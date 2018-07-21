import os, sys

import subprocess
import tempfile
import shutil
import json
import ast
import ConfigSpace as CS


def load_data(dest_dir, include_learning_rate=False):
    '''
    Training_loss, validation_loss, training_accuracy, validation_accuracy for the last epoch
    '''
    
    info = {}
    
    with open(os.path.join(dest_dir,'results.txt'), 'r') as fh:
        data = [ast.literal_eval(json.loads(line)) for line in fh.readlines()]
        
    with open(os.path.join(dest_dir,'log.txt'), 'r') as fh:
        info['config'] = '\n'.join(fh.readlines())
        
    if include_learning_rate:
        with open(os.path.join(dest_dir,'lr.txt'), 'r') as fh:
            info['learning_rates'] = [json.loads(line)['learningRate'] for line in fh.readlines()]
            
    info['loss'] = [d['train_loss'] for d in data]
    info['error'] = [d['train_top1'] for d in data]
    info['val_error'] = [d['valid_top1'] for d in data]
    info['test_error'] = [d['test_top1'] for d in data]
    
    return(info)



def configuration_cifar10(config, budget, min_budget, eta, config_id, directory, source=''):
	'''
	Uses code in arch_space to train with the given config
	'''
	dest_dir = os.path.join(directory, "_".join(map(str, config_id)))

	ret_dict =  { 'loss': float('inf'), 'info': None}

	try:
	    bash_strings = ["cd %s; python main.py --dataset cifar10 --num_threads %i"%(torch_source, 1), 
	                    "--batch_size {batch_size}".format(**config),
	                    "--weight_decay {weight_decay}".format(**config),
	                    "--LR {learning_rate} --momentum {momentum}".format(**config),
	                    "--save {} --budget {}".format(dest_dir, budget),
                	    "--apply_shakeShake --apply_shakeDrop --cutout",
                	    "--length {length} --alpha {alpha}".format(**config),
                	    "--death_rate {death_rate}".format(**config),
                	    "--T_e {} --T_mul {}".format(min_budget, 2.0),
                	    "--arch macro_model --lr_shape cosine", 
                	    "--forward_shake --backward_shake",
                	    "--shake_image",
                	    "--nr_main_blocks {nr_main_blocks} --nr_convs {nr_convs}".format(**config),
                	    "--nr_residual_blocks_1 {nr_residual_blocks_1}".format(**config),
                	    "--nr_residual_blocks_2 {nr_residual_blocks_2}".format(**config),
                	    "--nr_residual_blocks_3 {nr_residual_blocks_3}".format(**config),
                	    "--initial_filters {initial_filters}".format(**config),
                	    "--widen_factor_1 {widen_factor_1}".format(**config),
                	    "--widen_factor_2 {widen_factor_2}".format(**config),
                	    "--widen_factor_3 {widen_factor_3}".format(**config),
                	    "--res_branches_1 {res_branches_1}".format(**config),
                	    "--res_branches_2 {res_branches_2}".format(**config),
                	    "--res_branches_3 {res_branches_3}".format(**config),]
	    
	    subprocess.check_call( " ".join(bash_strings), shell=True)
	    info = load_data(dest_dir)
	    ret_dict = { 'loss': info['val_error'][-1], 'info': info}
	
	except:
	    print("Entering exception!!")
	    raise
	
	return (ret_dict)
