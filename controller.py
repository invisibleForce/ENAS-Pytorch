"""
child
spec:
    Class Controller is implemented.
func list:
    __init__
    train - train the controller
        controller model
    
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
# home made
from controller_model import ControllerModel
from child import Child
from data_utils import read_data
from utils import print_sample_arch, display_sample_arch
DEBUG = 0

class Controller(nn.Module):
    """
    Controller class.
    It describes how to train a controller
        1) train
    """
    def __init__(self,
               device='gpu',
               lstm_size=32,
               lstm_num_layers=2,
               child_num_layers=6,
               num_op=4,
               train_step_num=50,
               ctrl_batch_size=20,
               opt_algo='adam',
               lr_init=0.00035,
               lr_gamma=0.1,
               temperature=5,
               tanh_constant=2.5,
               entropy_weight=0.0001,
               baseline_decay=0.999,
               skip_target=0.4,
               skip_weight=0.8
              ):
        """
        1. init params
        2. create a graph which contains the sampled subgraph
        """
        super(Controller, self).__init__() # init the parent class of Net, i.e., nn.Module
        # config of controller model
        # child model
        self.child_num_layers = child_num_layers # imgs of dataset
        # ctrl model
        self.lstm_size = lstm_size # labels of dataset 
        self.lstm_num_layers = lstm_num_layers # number of classes
        self.num_op = num_op # 
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.skip_target = skip_target
        # ctrl training
        self.ctrl_batch_size=ctrl_batch_size
        self.opt_algo=opt_algo
        self.lr_init = lr_init
        self.lr_gamma = lr_gamma
        self.train_step_num = train_step_num
        self.entropy_weight = entropy_weight
        self.baseline_decay = baseline_decay
        self.skip_weight = skip_weight
        # device
        self.device = device
        # # training parameters on cpu
        if self.device == 'gpu':
            self.reward = torch.zeros(1).cuda() # rewards of samples
            self.baseline = torch.zeros(1).cuda() # base line
            self.log_prob = torch.zeros(1).cuda() # log_probs of samples
            self.entropy = torch.zeros(1).cuda() # entropys of samples
            # self.skip_rate = torch.zeros(1) # skip_rates of samples
            self.skip_penalty = torch.zeros(1).cuda() # skip_penaltys of samples
            self.loss = torch.zeros(1).cuda() # loss
        else:
            self.reward = torch.zeros(1) # rewards of samples
            self.baseline = torch.zeros(1) # base line
            self.log_prob = torch.zeros(1) # log_probs of samples
            self.entropy = torch.zeros(1) # entropys of samples
            # self.skip_rate = torch.zeros(1) # skip_rates of samples
            self.skip_penalty = torch.zeros(1) # skip_penaltys of samples
            self.loss = torch.zeros(1) # loss
        # training parameters on gpu
        

        # build controller
        self.ctrl = ControllerModel(child_num_layers=child_num_layers,
               lstm_size=lstm_size,
               lstm_num_layers=lstm_num_layers,
               num_op=num_op,
               temperature=temperature,
               tanh_constant=tanh_constant,
               skip_target=skip_target,
               device=device)
        # Optimizer; use SGD
        if DEBUG: print('#param', len(list(self.ctrl.parameters())))
        # style: Adam
        # self.optimizer = optim.Adam(self.ctrl.parameters(), lr=self.lr_init, betas=(0, 0.999)) # ENAS code sets beta1=0
        self.optimizer = optim.Adam(self.ctrl.parameters(), lr=self.lr_init)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr_init, weight_decay=self.l2_reg, momentum=0.9, nesterov=True)
        
        # learning rate scheduler - not mentioned in paper, not use it first
        # style: exponential decaying
        # lr = gamma * lr for each epoch
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_gamma)
        # style: multistepLR
        # decay lr every step_size epochs
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1,2], gamma=0.1)
        # style: stepLR; 
        # decay lr every step_size epochs
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
        # style: cosine
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.lr_cos_Tmax, eta_min=self.lr_cos_lmin)
    
    def train_epoch(self, child_model, images, labels, file):    
        """
        train controller for an epoch
        Procedure
        for N train stpes:
            for M child archs: - like obtain a batch of data
                sample a child architecture
                validate the sampled arch on a single minibatch of validation set
                obtain reward
                add weighed entropy to reward
                update baseline
                    exponential moving average of previous rewards
                cal loss: add weighed skip penalty to loss
            loss = avg( sample_log_prob * (reward - baseline) + skip_weight * skip_penaltys )
            zero grads
            cal grapds
                loss.backward() = REINFORCE
            update params of ctrl
            
        Args:
            
        Return:
            
        """
        # rewards = []
        # skip_rates = []
        # log_probs = []
        # entropys = []
        # skip_penaltys = []
        for step in range(self.train_step_num):
            # a single step of training
            # sample a batch of child archs and obtain their metrics
            if self.device == 'gpu':
                loss = torch.zeros(self.ctrl_batch_size).cuda()
            else:
                loss = torch.zeros(self.ctrl_batch_size)
            t_step = time.time()
            for sample_cnt in range(self.ctrl_batch_size):
                # sample a child arch
                self.ctrl.net_sample()
                # valid a sampled arch and obtain reward
                self.reward = child_model.valid_rl(self.ctrl.sample_arch, images, labels) 
                # add weighed entropy to reward
                self.entropy = self.ctrl.sample_entropy
                self.reward += self.entropy_weight * self.entropy
                # update baseline
                with torch.no_grad():
                    self.baseline = self.baseline + (1 - self.baseline_decay) * (self.reward - self.baseline)
                # update loss
                self.log_prob = self.ctrl.sample_log_prob
                self.skip_penalty = self.ctrl.sample_skip_penaltys
                loss[sample_cnt] = self.log_prob * (self.reward - self.baseline) + self.skip_weight * self.skip_penalty
                # # cal skip rate and append it
                # skip_rate = self.ctrl.sample_skip_count
                # normalize = self.child_num_layers * (self.child_num_layers - 1) / 2
                # skip_rate /= normalize
                # skip_rates.append(skip_rate)
            self.loss = loss.sum() / self.ctrl_batch_size # avg loss
            # zero grads
            self.optimizer.zero_grad()
            # cal grads
            # self.loss.backward(retain_graph=True)
            self.loss.backward()
            # update weights
            self.optimizer.step()
            # print(self.ctrl.net['op_fc'].weight.grad) # check grad is updated
            # cal time consumed per step
            t_step = time.time() - t_step
            if step % 10 == 0:
                print('step', step)
                display_sample_arch(self.ctrl.sample_arch)
                file.write('step:'+str(step))
                print_sample_arch(self.ctrl.sample_arch, file)
                print('time_per_step', t_step)
                file.write('time_per_step'+str(t_step))

    def get_op_portion(self):
        """
        Count number of each type of ops in the sample arch
            
        Args: sample_arch
        Return:
        """
        op_counts = [0] * self.num_op
        sample_arch = self.ctrl.sample_arch
        for i in range(self.child_num_layers):
            op_counts[sample_arch[2 * i][0]] += 1

        return op_counts

    def get_op_percent(self, op_histroy):
        """
        Avg number of each type of ops in the sample arch
            
        Args: op_histroy
        Return:
        """
        num_samples = len(op_histroy)
        op_histroy = np.stack(op_histroy)
        op_history = np.sum(op_histroy, axis=0)
        op_history = op_history / np.sum(op_history) # portion of each type of op

        return op_history

    def eval(self, child_model, arc_num, images, labels, file):
        """
        evaluate controller using validating data set.
        It samples several archs and validate them on 
        the whole validate set.
            
        Args:
            
        Return:
            
        """
        accuracy = []
        arcs = []
        op_percent = []
        for _ in range(arc_num):
            # sample a child arch
            self.ctrl.net_sample()
            arcs.append(self.ctrl.sample_arch)
            # valid a sampled arch and obtain reward
            eval_acc = child_model.eval(self.ctrl.sample_arch, images, labels) 
            accuracy.append(eval_acc)
            # get the op analysis
            op_percent.append(self.get_op_portion())
        # obtain averaged op_history
        op_percent = self.get_op_percent(op_percent)
        # print to file  
        # accuracy      
        file.write('arch \t accuracy\n')    
        for i, acc in enumerate(accuracy):
            file.write('%d \t %f\n' % (i, acc)) 
        # arch       
        for i, arc in enumerate(arcs):    
            file.write('arch#: %d\n' % i)
            print_sample_arch(arc, file)
        
        return accuracy, op_percent

    def derive_best_arch(self, child_model, arc_num, images, labels, file):
        """
        derive the final child model using controller
        procedure
            1. sample 1000 archs
            2. test them on test data set
            3. select the one with highest accuracy as the best arch
        Args:
            
        Return:
            best_arch
        """
        accuracy = []
        arcs = []
        best_arch = []
        best_accuracy = 0
        for _ in range(arc_num):
            # sample a child arch
            self.ctrl.net_sample()
            arcs.append(self.ctrl.sample_arch)
            # valid a sampled arch and obtain reward
            eval_acc = child_model.eval(self.ctrl.sample_arch, images, labels) 
            accuracy.append(eval_acc)
            # select the best arch
            if eval_acc > best_accuracy:
                best_accuracy = eval_acc
                best_arch = self.ctrl.sample_arch
        
        # print to file  
        # best accuracy and arc
        file.write('best accuracy: %f\n' % best_accuracy)
        file.write('best arch \n')
        print_sample_arch(best_arch, file)
        # accuracy    
        file.write('-' * 30 + '\n')   
        file.write(' accuracies \n')   
        file.write('-' * 30 + '\n')   
        file.write('arch \t accuracy\n')    
        for i, acc in enumerate(accuracy):
            file.write('%d \t %f\n' % (i, acc)) 
        # arch     
        file.write('-' * 30 + '\n')   
        file.write(' archs \n')   
        file.write('-' * 30 + '\n')     
        for i, arc in enumerate(arcs):    
            file.write('arch#: %d\n' % i)
            print_sample_arch(arc, file)

        return best_accuracy, best_arch

def test_ctrl():
    # obtain datasets
    t = time.time()
    images, labels = read_data()
    t = time.time() - t
    print('read dataset consumes %.2f sec' % t)
    # config of a model
    class_num = 10
    child_num_layers = 6
    out_channels = 32
    batch_size = 32
    device = 'gpu'
    epoch_num = 4
    # files to print sampled archs
    child_filename = 'child_file.txt'
    ctrl_filename = 'controller_file.txt'
    child_file = open(child_filename, 'w')
    ctrl_file = open(ctrl_filename, 'w')
    # create a controller
    ctrl = Controller(child_num_layers=child_num_layers)
    # create a child, set epoch to 1; later this will be moved to an over epoch
    child = Child(images, labels, class_num, child_num_layers, out_channels, batch_size, device, 1)
    print(len(list(child.net.graph)))
    # print(child.net.graph)
    # train multiple epochs
    for _ in range(epoch_num):
        # sample an arch
        ctrl.ctrl.net_sample()
        sample_arch = ctrl.ctrl.sample_arch
        print_sample_arch(sample_arch, child_file)
        # train a child model
        # t = time.time()
        # child.train(sample_arch)
        # t = time.time() - t
        # print('child training time %.2f sec' % t)

        # train controller
        t = time.time()
        ctrl.train(child, ctrl_file)
        t = time.time() - t
        print('ctrller training time %.2f sec' % t)

# ------------------
# Testbench
# ------------------
if __name__ == '__main__':
    test_ctrl()