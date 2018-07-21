import os
import sys
import json
import time
import torch
import importlib
import logging
import numpy as np

sys.path.insert(0, os.getcwd())

from train import Trainer
from opts import Parser
from dataloader import Loader
from checkpoints import save_checkpoint, load_latest

from torch.optim.lr_scheduler import MultiStepLR
from util.SGDR_time import CosineAnnealingRestartsLR

#from model.resnet import ResNet18
from model.utils import load_model

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)

def log(stats, logfile):
    with open(logfile, 'a') as file:
        json.dump('json_stats: ' + str(stats), file)
        file.write('\n')

def main():
    parser = Parser()
    config = parser.config

    for param, value in config.__dict__.items():
        print(param + '.' * (50 - len(param) - len(str(value))) + str(value))
    print()

    # Load previous checkpoint if it exists
    checkpoint = load_latest(config)

    # Create model
    model = load_model(config, checkpoint)   

    # print number of parameters in the model
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print('Total number of parameters: \33[91m{}\033[0m'.format(n_params))
    
    # Load train and test data
    train_loader, valid_loader, test_loader = Loader(config)

    n_batches = int(len(train_loader.dataset.train_data) / config.batch_size) 

    # save the configuration
    with open(os.path.join(config.save, 'log.txt'), 'w') as file:
        json.dump('json_stats: ' + str(config.__dict__), file)

    # Instantiate the criterion, optimizer and learning rate scheduler
    criterion = torch.nn.CrossEntropyLoss(size_average=True)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LR,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov
    )

    start_time = 0
    if checkpoint is not None:
        start_epoch = checkpoint['time'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])

    if config.lr_shape == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=[81, 122], gamma=0.1)
    elif config.lr_shape == 'cosine':
        if checkpoint is not None:
            scheduler = checkpoint['scheduler']
        else:
            scheduler = CosineAnnealingRestartsLR(optimizer, 1, config.T_e, T_mul=config.T_mul)

    # The trainer handles the training loop and evaluation on validation set
    trainer = Trainer(model, criterion, config, optimizer, scheduler)

    epoch = 1

    while True:
        # Train for a single epoch
        train_top1, train_loss, stop_training = trainer.train(epoch, train_loader)
        
        # Run model on the validation and test set
        valid_top1 = trainer.evaluate(epoch, valid_loader, 'valid')
        test_top1 = trainer.evaluate(epoch, test_loader, 'test')

        current_time = time.time()

        results = {'epoch': epoch,
                   'time': current_time,
                   'train_top1': train_top1,
                   'valid_top1': valid_top1,
                   'test_top1': test_top1,
                   'train_loss': float(train_loss.data),
                   }

        with open(os.path.join(config.save, 'results.txt'), 'w') as file:
            json.dump(str(results), file)
            file.write('\n')

        print('==> Finished epoch %d (budget %.3f): %7.3f (train) %7.3f (validation) %7.3f (test)' % (epoch, config.budget, train_top1, valid_top1, test_top1))

        if stop_training:
            break

        epoch += 1

    if start_time >= config.budget:
        trainer.evaluate(epoch, test_loader, 'test')
    else:
        save_checkpoint(int(config.budget), trainer.model, trainer.optimizer, trainer.scheduler, config)

if __name__ == '__main__':
    main()
