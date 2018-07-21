import os
import time
import pickle
import argparse
import logging

from workers.cifar10_worker import CIFAR10_base as worker

from hpbandster.optimizers.bohb import BOHB
from hpbandster.core.nameserver import NameServer
from hpbandster.utils import *
import hpbandster.core.result as hputil

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s',  datefmt='%H:%M:%S')

parser = argparse.ArgumentParser(description='Run BOHB on CIFAR10 search space.')

parser.add_argument('--dest_dir', type=str, help='the destination directory. A new subfolder is created for each benchmark/dataset.', default='./data/')
parser.add_argument('--num_iterations', type=int, help='number of Hyperband iterations performed.', default=64)
parser.add_argument('--run_id', type=int, default=0)
parser.add_argument('--working_directory', type=str, help='directory where to store the live rundata', default=None)
parser.add_argument('--array_id', type=int, default=1)
parser.add_argument('--total_num_workers', type=int, default=20)
parser.add_argument('--min_budget', type=int, default=400, help='minimum budget given to HyperBand (in seconds).')
parser.add_argument('--max_budget', type=int, default=10800, help='maximum budget given to HyperBand (in seconds).')
parser.add_argument('--eta', type=int, default=3, help='Multiplicative factor accross budgets.')

args = parser.parse_args()

min_budget = args.min_budget
max_budget = args.max_budget
eta = args.eta

if args.array_id == 1:
	os.makedirs(args.working_directory, exist_ok=True)
	
	NS = NameServer(run_id=args.run_id, nic_name='eth0', working_directory=args.working_directory)
	ns_host, ns_port = NS.start()

	# BOHB is usually so cheap, that we can affort to run a worker on the master node, too.
	worker = worker(min_budget=min_budget, max_budget=max_budget, eta=eta,
					nameserver=ns_host, nameserver_port=ns_port, run_id=args.run_id)
	worker.run(background=True)

	#instantiate BOHB and run it
	result_logger = hputil.json_result_logger(directory=args.working_directory, overwrite=True)
	
	HPB = BOHB(configspace = worker.get_config_space(),
               working_directory=args.working_directory,
               run_id = args.run_id,
               eta=eta,min_budget=min_budget, max_budget=max_budget,
               host=ns_host,
               nameserver=ns_host,
               nameserver_port = ns_port,
               ping_interval=3600,
               result_logger=result_logger
               )
	
	res = HPB.run(n_iterations = args.num_iterations,
	              min_n_workers = args.total_num_workers  # BOHB can wait until a minimum number of workers is online before starting
	              )	
	
	with open(os.path.join(args.working_directory , 'results.pkl'), 'wb') as fh:
		pickle.dump(res, fh)
	
	HPB.shutdown(shutdown_workers=True)
	NS.shutdown()

else:
	time.sleep(30)
	
	host = nic_name_to_host('eth0')

	worker = worker(min_budget=min_budget, max_budget=max_budget, eta=eta,
					host=host, run_id=args.run_id)
	
	worker.load_nameserver_credentials(args.working_directory)
	worker.run(background=False)
