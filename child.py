"""
child
spec:
    Class Child is implemented.
func list:
    __init__
    build_train
        model
    build_valid
        model
    build_test
        model
    build_valid_rl
        model
log
1. train nas model on gpu
"""
# packages
# std
import os
import sys
import time
# installed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim # optimizer
import random
# home made
from child_model import ChildModel
from data_utils import read_data, augment

DEBUG = 0

class Child(nn.Module):
    """
    Child class.
    It describes functions for
        1) training a child
        def build_train:
        2) validating a child
        def build_valid
        3) testing a child
        def build_test
        4) validating a child for RL
        def build_valid_rl
    """
    def __init__(self,
               class_num,
               num_layers=6,
               out_channels=24,
               batch_size=32,
               device='gpu', 
               lr_init=0.05,
               lr_gamma=0.1,
               lr_cos_lmin=0.001,
               lr_cos_Tmax=2,
               l2_reg=1e-4,
               run_loss_every=100
              ):
        """
        1. init params
        2. create a graph which contains the sampled subgraph
        """
        super(Child, self).__init__() # init the parent class of Net, i.e., nn.Module
        # data set used for training, validating, testing
        self.class_num = class_num # number of classes
        # parameters for building a child model
        self.num_layers = num_layers # 
        self.out_channels = out_channels
        # parameters used for training a child model
        # batch size (for training)
        self.batch_size = batch_size
        self.run_loss_every = run_loss_every
        # optimizer
        self.l2_reg = l2_reg
        # learning rate
        self.lr_init = lr_init
        self.lr_gamma = lr_gamma
        self.lr_cos_lmin = lr_cos_lmin
        self.lr_cos_Tmax = lr_cos_Tmax
        # device
        self.device = device
        # build D\AG = net
        self.net = ChildModel(class_num, num_layers, out_channels)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # Optimizer; use SGD
        if DEBUG: print('#param', len(list(self.net.parameters())))
        # style: Nesterov momentum
        # l2_reg = weight_decay
        self.optimizer = optim.SGD([{'params': self.net.parameters(), 'initial_lr': self.lr_init}], lr=self.lr_init, weight_decay=self.l2_reg, momentum=0.9, nesterov=True)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr_init, weight_decay=self.l2_reg, momentum=0.9, nesterov=True)
        
        # learning rate scheduler
        # style: exponential decaying - abandon
        # lr = gamma * lr for each epoch
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_gamma)
        # style: multistepLR
        # decay lr every step_size epochs
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1,2], gamma=0.1)
        # style: stepLR; 
        # decay lr every step_size epochs
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
        # style: cosine
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.lr_cos_Tmax, eta_min=self.lr_cos_lmin)
    
    def get_batch(self, images, labels, step):
        """
        get a batch of data
        """
        # obtain a batch
        batch_size = self.batch_size
        batch_images = images[step * batch_size : (step + 1) * batch_size] 
        batch_labels = labels[step * batch_size : (step + 1) * batch_size] 
        if DEBUG: print('get_batch', type(batch_images))
        if DEBUG: print('get_batch', type(batch_labels))
        # augment images
        batch_images = augment(batch_images)
        # transfer batch images and labels to tensor
        # numpy to tensor
        # batch_images = torch.from_numpy(batch_images)
        # batch_labels = torch.from_numpy(batch_labels)
        # if DEBUG: print(type(batch_images))
        # if DEBUG: print(type(batch_labels))
        # convert batch labels from int32 to int64
        # if DEBUG: print('get_batch', batch_labels.dtype)
        # batch_labels = batch_labels.long()
        # if DEBUG: print('get_batch', batch_labels.dtype)

        return batch_images, batch_labels

    def train_epoch(self, sample_arch, images, labels, epoch, train_step):    
        """
        train a sampled child model on an epoch
        Args:
            epoch: number of epochs; default = 1
            sample_arch: a list consisting of 2 * num_layers elements
                op_id = sample_arch[2k]: operation id
                skip = sample_arch[2k + 1]: element i of such abinary vector 
                    is used to describe whether the previous layer i is used 
                    as an input
        Return:
            
        """
        # track running loss
        running_loss = 0.0
        print('lr=', self.scheduler.get_lr())
        # print('Epoch', epoch + 1, 'lr=', self.lr_init)
        for step in range(train_step): # 
            # get a minibatch for cpu or gpu
            batch_inputs, batch_labels = self.get_batch(images, labels, step)
            # zeros the grads for weights
            self.optimizer.zero_grad()
            # forward
            outputs = self.net.model(batch_inputs, sample_arch)
            # loss
            loss = self.criterion(outputs, batch_labels)
            # backward
            loss.backward()
            # update weights
            self.optimizer.step()
            
            # update running loss
            running_loss += loss.item()
            if step % self.run_loss_every == (self.run_loss_every - 1):
                print('[%d, %5d], loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / self.run_loss_every))
                running_loss = 0.0
        
        # decay learning rate
        self.scheduler.step()
        # print(self.scheduler.get_lr()) # check lr updated

    def valid_rl(self, sample_arch, images, labels):    
        """
        validate a sampled child model on a random minibatch of validation set
        used to generate reward of controller
        Args:
            sample_arch: a child arch sample
        Return: 
            valid_acc: accuracy
        """
        # validating
        # get a minibatch for cpu or gpu
        high = labels.size()[0] // self.batch_size
        
        batch_idx = torch.randint(high, (1,1))
        # batch_idx = random.randint(0, high)
        batch_inputs, batch_labels = self.get_batch(images, labels, batch_idx)
        
        # forward
        outputs = self.net.model(batch_inputs, sample_arch)
        
        # cal accuracy
        value, idx = torch.topk(outputs, 1)
        idx = idx.reshape((-1))
        accuracy = (idx == batch_labels).float().sum()
        accuracy /= self.batch_size
        
        return accuracy

    def eval(self, sample_arch, images, labels):    
        """
        evaluate a sampled child model on a given dataset
        Args:
            sample_arch: a child arch sample
        Return: 
            valid_acc: accuracy
        """
        step_num = labels.size()[0] // self.batch_size
        total_accuracy = 0
        for i in range(step_num):
            # get a minibatch for cpu or gpu
            batch_inputs, batch_labels = self.get_batch(images, labels, i)
            # forward
            outputs = self.net.model(batch_inputs, sample_arch)
            # accumulate hit predictions
            _, idx = torch.topk(outputs, 1)
            idx = idx.reshape((-1))
            total_accuracy += (idx == batch_labels).float().sum() # count the correct prediction
        # total accuracy
        total_accuracy /= (step_num * self.batch_size)
        
        return total_accuracy

     

def test_child():
    # obtain datasets
    t = time.time()
    images, labels = read_data()
    t = time.time() - t
    print('read dataset consumes %.2f sec' % t)
    # config of a model
    class_num = 10
    num_layers = 6
    out_channels = 32
    batch_size = 32
    device = 'gpu'
    epoch_num = 4
    # sample a child model
    sample_arch = []
    # layer 0
    sample_arch.append([0]) # op, c3
    sample_arch.append([]) # skip, none
    # layer 1
    sample_arch.append([1]) # op, c5
    sample_arch.append([1]) # skip=layer i + 1 input, l0=1
    # layer 2
    sample_arch.append([3]) # op, mp
    sample_arch.append([0, 0]) # skip=layer i + 1 input, l0=0, l1=0
    # layer 3
    sample_arch.append([1]) # op, c5
    sample_arch.append([1, 0, 1]) # skip=layer i + 1 input, l0=1, l1=0, l2=1
    # layer 4
    sample_arch.append([0]) # op, c3
    sample_arch.append([0, 0, 0, 0]) # skip=layer i + 1 input, l0=0, l1=0, l2=0, l3=0
    # layer 5
    sample_arch.append([2]) # op, ap
    sample_arch.append([0, 0, 0, 0, 0]) # skip=layer i + 1 input, l0=0, l1=0, l2=0, l3=0
    print(sample_arch)
    
    # create a child
    child = Child(images, labels, class_num, num_layers, out_channels, batch_size, device, epoch_num)
    print(len(list(child.net.graph)))
    # print(child.net.graph)
    # train a child model
    t = time.time()
    child.train(sample_arch)
    t = time.time() - t
    print('training time %.2f sec' % t)

    # # train another sample_arch
    # sample_arch = []
    # # layer 0
    # sample_arch.append([1]) # op, c5
    # sample_arch.append([]) # skip, none
    # # layer 1
    # sample_arch.append([0]) # op, c3
    # sample_arch.append([1]) # skip=layer i + 1 input, l0=1
    # # layer 2
    # sample_arch.append([3]) # op, mp
    # sample_arch.append([1, 0]) # skip=layer i + 1 input, l0=1, l1=0
    # # layer 3
    # sample_arch.append([0]) # op, c3
    # sample_arch.append([1, 0, 1]) # skip=layer i + 1 input, l0=1, l1=0, l2=1
    # # layer 4
    # sample_arch.append([0]) # op, c3
    # sample_arch.append([0, 1, 0, 1]) # skip=layer i + 1 input, l0=0, l1=1, l2=0, l3=1
    # # layer 5
    # sample_arch.append([2]) # op, ap
    # sample_arch.append([0, 0, 0, 1, 1]) # skip=layer i + 1 input, l0=0, l1=0, l2=0, l3=1, l4=1
    # print(sample_arch)

    # print(len(list(child.net.graph)))
    # print(child.net.graph)
    # train a child model
    t = time.time()
    child.train(sample_arch)
    t = time.time() - t
    print('training time %.2f sec' % t)

# ------------------
# Testbench
# ------------------
if __name__ == '__main__':
    test_child()