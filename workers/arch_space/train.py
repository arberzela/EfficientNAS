import time
import os
import torch

from torch.autograd import Variable

from util.transforms import mixup_data, mixup_criterion
from checkpoints import save_checkpoint

class Trainer(object):
    def __init__(self, model, criterion, config, optimizer, scheduler):
        self.model = model.cuda()
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.config = config

        self.cumulative_time = 0

        self.use_cuda = torch.cuda.is_available()

    def train(self, epoch, train_loader):
        '''
            Trains the model for a single epoch
        '''

        train_size = int(0.9 * len(train_loader.dataset.train_data) / self.config.batch_size)
        top1_sum, loss_sum, last_loss = 0.0, 0.0, 0.0
        N = 0

        print('\33[1m==> Training epoch # {}\033[0m'.format(str(epoch)))
        
        self.model.train()

        start_time_step = time.time()
        for step, (data, targets) in enumerate(train_loader):
            data_timer = time.time()
   
            if self.use_cuda:
                data = data.cuda()
                targets = targets.cuda()

            data, targets_a, targets_b, lam = mixup_data(data, targets, self.config.alpha, self.use_cuda)

            data = Variable(data)
            targets_a = Variable(targets_a)
            targets_b = Variable(targets_b)
            
            batch_size = data.size(0)

            if epoch != 1 or step != 0:
                self.scheduler.step(epoch=self.scheduler.cumulative_time)
            else:
                self.scheduler.step()

            # used for SGDR with seconds as budget
            start_time_batch = time.time()

            self.optimizer.zero_grad()

            outputs = self.model(data)
            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(self.criterion, outputs)
            loss.backward()

            self.optimizer.step()

            # used for SGDR with seconds as budget
            delta_time = time.time() - start_time_batch
            self.scheduler.cumulative_time += delta_time
            self.cumulative_time += delta_time
            self.scheduler.last_step = self.scheduler.cumulative_time - delta_time - 1e-10

            # Each time before the learning rate restarts we save a checkpoint in order to create snapshot ensembles after the training finishes
            if (epoch != 1 or step != 0) and (self.cumulative_time > self.config.T_e + delta_time + 5) and (self.scheduler.last_step < 0):
                save_checkpoint(int(round(self.cumulative_time)), self.model, self.optimizer, self.scheduler, self.config)

            top1 = self.compute_score_train(outputs, targets_a, targets_b, batch_size, lam)
            top1_sum += top1 * batch_size
            last_loss = loss
            loss_sum += loss * batch_size
            N += batch_size

            #print(' | Epoch: [%d][%d/%d]   Time %.3f  Data %.3f  Err %1.3f  top1 %7.2f  lr %.4f' 
            #        % (epoch, step + 1, train_size, self.cumulative_time, data_timer - start_time_step, loss.data, top1, self.scheduler.get_lr()[0]))

            start_time_step = time.time()

            if self.cumulative_time >= self.config.budget:
                print(' * Stopping at Epoch: [%d][%d/%d] for a budget of %.3f s' % (epoch, step + 1, train_size, self.config.budget))
                return top1_sum / N, loss_sum / N, True

        return top1_sum / N, loss_sum / N, False


    def evaluate(self, epoch, test_loader, mode):

        if mode == 'test':
            test_size = int(len(test_loader.dataset.test_data) / self.config.batch_size)
        elif mode == 'valid':
            test_size = int(self.config.valid_frac * len(test_loader.dataset.train_data) / self.config.batch_size)

        top1_sum = 0.0
        N = 0
        
        self.model.eval()

        start_time_step = time.time()
        for step, (data, targets) in enumerate(test_loader):
            data_timer = time.time()
   
            if self.use_cuda:
                data = Variable(data.cuda(), volatile=True)
                targets = Variable(targets.cuda(), volatile=True)
            else:
                data = Variable(data, volatile=True)
                targets = Variable(targets, volatile=True)
            
            batch_size = data.size(0)

            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            top1 = self.compute_score_test(outputs, targets, batch_size)
            top1_sum += top1 * batch_size
            N += batch_size

            #if mode == 'test':
            #    print(' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)' % (
            #          epoch, step + 1, test_size, time.time() - start_time_step, data_timer - start_time_step, top1, top1_sum / N))

            #elif mode == 'valid':
            #    print(' | Valid: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)' % (
            #          epoch, step + 1, test_size, time.time() - start_time_step, data_timer - start_time_step, top1, top1_sum / N))

            start_time_step = time.time() 

        self.model.train()

        #if mode == 'valid':
        #    print('\33[1m * Finished epoch # %d     top1: \33[91m%7.3f\033[0m\n' % (epoch, top1_sum / N))
        #elif mode == 'test':
        #    print('\33[1m * Final error on the test set     top1: \33[91m%7.3f\033[0m\n' % (top1_sum / N))
            
        return top1_sum / N
    

    def compute_score_train(self, output, target_a, target_b, batch_size, lam):
        pred = torch.max(output.data, 1)[1]
        top1 = lam * pred.eq(target_a.data).sum() + (1 - lam) * pred.eq(target_b.data).sum()
        top1 = 1 - top1 / batch_size

        return top1 * 100

    def compute_score_test(self, output, target, batch_size):
        pred = torch.max(output.data, 1)[1]
        top1 = pred.eq(target.data).sum()
        top1 = 1 - top1 / batch_size

        return top1 * 100


