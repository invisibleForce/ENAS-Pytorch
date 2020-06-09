"""
child
spec:
    Class Controller is implemented.
func list:
    __init__
    train - train the controller
        controller model
    
log
6.3
1. train nas model on gpu
6.5
1. add final arch deriving
6.6 
1. add perf metrics tracking
    a. child valid accuracy
    b. controller average valid accuracy
    c. final child validation
    d. operation distribution of controller validation
"""
# packages
# std
import os
import sys
import time
# installed
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim # optimizer
# home made
from controller import Controller
from child import Child
from data_utils import read_data
from utils import print_sample_arch, display_sample_arch
from config import enas_cfg
DEBUG = 0

def main():
    train(enas_cfg)

def train(config):
    """
    Main entrance of the enas
    It describes the procedure used to run a complete arch search.
    """
    # ===================
    # Config
    # ===================
    # -----------
    # platform 
    # -----------
    platform = config.device

    # -----------
    # enas 
    # -----------
    epoch_num = config.epoch_num
    retrain_epoch_num = config.retrain_epoch_num
    # -----------
    # child 
    # -----------
    # model
    child_class_num = config.child_class_num
    child_num_layers = config.child_num_layers
    child_out_channels = config.child_out_channels
    child_num_op = config.child_num_op
    # --- training
    child_data_path = config.child_data_path
    child_num_valids = config.child_num_valids
    child_batch_size = config.child_batch_size
    child_opt_algo = config.child_opt_algo # not used indeed
    # learning scheduler = cosine anealing
    child_lr_init = config.child_lr_init
    child_lr_gamma = config.child_lr_gamma
    child_lr_cos_lmin = config.child_lr_cos_lmin
    child_lr_cos_Tmax = config.child_lr_cos_Tmax
    # weight decay = l2 regularization
    child_l2_reg = config.child_l2_reg
    # optimizer = sgd + Nestrov momentum
    # log
    child_run_loss_every = config.child_run_loss_every
    # valid
    child_valid_every_epochs = config.child_valid_every_epochs
    # -----------
    # controller 
    # -----------
    # model
    ctrl_lstm_size = config.ctrl_lstm_size
    ctrl_lstm_num_layers = config.ctrl_lstm_num_layers
    # --- training
    ctrl_train_step_num = config.ctrl_train_step_num # number of training steps per epoch
    ctrl_batch_size = config.ctrl_batch_size # number of samples per training step
    ctrl_opt_algo = config.ctrl_opt_algo # not used indeed
    ctrl_train_every_epochs = config.ctrl_train_every_epochs
    # learning scheduler = exponential decaying
    ctrl_lr_init = config.ctrl_lr_init
    ctrl_lr_gamma = config.ctrl_lr_gamma
    # baseline - reduce high variance; exponential moving average
    ctrl_baseline_decay = config.ctrl_baseline_decay
    # prevent from being permature of controller
    # applied to logits
    ctrl_temperature = config.ctrl_temperature
    ctrl_tanh_constant = config.ctrl_tanh_constant
    # add entropy to reward
    ctrl_entropy_weight = config.ctrl_entropy_weight
    # enforce skip sparsity 
    # add skip penalty to loss
    ctrl_skip_target = config.ctrl_skip_target
    ctrl_skip_weight = config.ctrl_skip_weight
    # validate
    ctrl_valid_every_epochs = config.ctrl_valid_every_epochs
    ctrl_eval_arc_num = config.ctrl_eval_arc_num
    ctrl_final_arc_num = config.ctrl_final_arc_num
    # -----------
    # output 
    # -----------
    child_filename = config.child_filename
    ctrl_filename = config.ctrl_filename
    final_child_filename = config.final_child_filename
    child_model_save_path = config.child_model_save_path
    ctrl_model_save_path = config.ctrl_model_save_path
    final_child_save_path = config.final_child_save_path
    # ===================
    # read datasets
    # ===================
    t = time.time()
    images, labels = read_data(child_data_path, child_num_valids)    # train, valid and test
    t = time.time() - t
    print('read dataset consumes %.2f sec' % t)
    
    
    # ===================
    # create nets
    # ===================
    # create a child, set epoch to 1; later this will be moved to an over epoch
    child = Child(
        class_num=child_class_num,
        num_layers=child_num_layers,
        out_channels=child_out_channels,
        batch_size=child_batch_size,
        device=platform, 
        lr_init=child_lr_init,
        lr_gamma=child_lr_gamma,
        lr_cos_lmin=child_lr_cos_lmin,
        lr_cos_Tmax=child_lr_cos_Tmax,
        l2_reg=child_l2_reg,
        run_loss_every=child_run_loss_every
    )
    print('layer num of a child:', len(list(child.net.graph)))
    
    # create a controller
    ctrl = Controller(
        device=platform,
        lstm_size=ctrl_lstm_size,
        lstm_num_layers=ctrl_lstm_num_layers,
        child_num_layers=child_num_layers,
        num_op=child_num_op,
        train_step_num=ctrl_train_step_num,
        ctrl_batch_size=ctrl_batch_size,
        opt_algo=ctrl_opt_algo,
        lr_init=ctrl_lr_init,
        lr_gamma=ctrl_lr_gamma,
        temperature=ctrl_temperature,
        tanh_constant=ctrl_tanh_constant,
        entropy_weight=ctrl_entropy_weight,
        baseline_decay=ctrl_baseline_decay,
        skip_target=ctrl_skip_target,
        skip_weight=ctrl_skip_weight)
    
    # ===================
    #  output files
    # ===================
    child_file = open(child_filename, 'w') 
    ctrl_file = open(ctrl_filename, 'w')
    final_child_file = open(final_child_filename, 'w') 
    
    # ===================
    #  gpu offloading
    # ===================
    # move net and data to gpu
    train_imgs = images['train'] 
    train_labels = labels['train'] 
    valid_imgs = images['valid']
    valid_labels = labels['valid']
    test_imgs = images['test']
    test_labels = labels['test']
    if platform == 'gpu': # check whether gpu is available or not
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        child.net.to(device) # move net to gpu
        train_imgs = train_imgs.cuda()
        train_labels = train_labels.cuda()
        valid_imgs = valid_imgs.cuda()
        valid_labels = valid_labels.cuda()
        test_imgs = test_imgs.cuda()
        test_labels = test_labels.cuda()
    
    train_step = int(train_imgs.size()[0] / child_batch_size)
    # ===================
    # training loop
    # ===================
    # perf metrics tracked
    child_valid_acc = []
    ctrl_valid_acc_avg = []
    op_percent_avg = []
    for epoch in range(epoch_num):
        print('Epoch', epoch)
        child_file.write('Epoch: %d \n' % (epoch))
        ctrl_file.write('Epoch: %d \n' % (epoch))
        # sample an arch
        print('---- sample an arch ----')
        ctrl.ctrl.net_sample()
        sample_arch = ctrl.ctrl.sample_arch
        child_file.write('---- sample an arch ----\n')
        print_sample_arch(sample_arch, child_file)
        
        # train a child model   
        print('---- train a child model ----')
        t = time.time()
        child.train_epoch(sample_arch, train_imgs, train_labels, epoch, train_step)
        t = time.time() - t
        print('child training time per epoch %.2f sec' % t)
        child_file.write('---- train a child model ----\n')
        child_file.write('child training time per epoch %.2f sec \n' % t)
        
        # validate a child model
        if (epoch + 1) % child_valid_every_epochs == 0:
            print('---- validate a child model ----')
            accuracy = child.eval(sample_arch, valid_imgs, valid_labels)
            child_valid_acc.append(accuracy)
            print('epoch: %d, accuracy: %f' % (epoch, accuracy))
            child_file.write('---- validate a child model ----\n') 
            child_file.write('epoch: %d, accuracy: %f\n' % (epoch, accuracy))
            
        
        # train controller
        if (epoch + 1) % ctrl_train_every_epochs == 0:
            print('---- train controller ----')
            t = time.time()
            ctrl.train_epoch(child, valid_imgs, valid_labels, ctrl_file)
            t = time.time() - t
            print('ctrller training time per epoch %.2f sec' % t)
            ctrl_file.write('---- train controller ----\n')
            ctrl_file.write('ctrller training time per epoch %.2f sec \n' % t)

        # validate controller
        if (epoch + 1) % ctrl_valid_every_epochs == 0:
            print('---- validate controller ----')
            ctrl_file.write('---- validate controller ----\n') 
            accuracy, op_percent = ctrl.eval(child, ctrl_eval_arc_num, valid_imgs, valid_labels, ctrl_file)
            acc_avg = torch.mean(torch.tensor(accuracy))
            ctrl_valid_acc_avg.append(acc_avg)
            op_percent_avg.append(op_percent)
            print('arch \t accuracy')
            for i, acc in enumerate(accuracy):
                print('%d \t %f' % (i, acc))

    # ===================
    # derive final child
    # ===================
    best_accuracy, best_arch = ctrl.derive_best_arch(child, ctrl_final_arc_num, test_imgs, test_labels, final_child_file)
    print('-------- best arch -------')
    display_sample_arch(best_arch)
    print('best accuracy', best_accuracy)
    # ===================
    # save models
    # ===================
    # save child model for reusing it
    # PATH = './enas_child.pth'
    torch.save(child.net.graph.state_dict(), child_model_save_path)
    # save ctrller for resuing it
    torch.save(ctrl.ctrl.net.state_dict(), ctrl_model_save_path)
    # ===================
    # plot controller training
    # ===================
    plot_metric(child_valid_acc, 'child_valid_acc', 'child_valid_acc')
    plot_metric(ctrl_valid_acc_avg, 'ctrl_valid_acc_avg', 'ctrl_valid_acc_avg')
    plot_stack_bar(op_percent_avg, 'op_percent_avg', 'op_percent_avg')
    # ===================
    # retrain final child
    # ===================
    # create an empty child
    final_child = Child(
        class_num=child_class_num,
        num_layers=child_num_layers,
        out_channels=child_out_channels,
        batch_size=child_batch_size,
        device=platform, 
        lr_init=child_lr_init,
        lr_gamma=child_lr_gamma,
        lr_cos_lmin=child_lr_cos_lmin,
        lr_cos_Tmax=retrain_epoch_num,
        l2_reg=child_l2_reg,
        run_loss_every=child_run_loss_every
    )

    if platform == 'gpu': # check whether gpu is available or not
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        final_child.net.to(device) # move net to gpu
    print(' --------- start retraining ---------')
    child_file.write( '--------- start retraining --------- \n')
    final_child_valid_acc = []
    t = time.time()
    for epoch in range(retrain_epoch_num):
        print('Epoch', epoch)
        child_file.write('Epoch: %d \n' % (epoch))
        final_child.train_epoch(best_arch, train_imgs, train_labels, epoch, train_step)
        # validate a child model
        if (epoch + 1) % child_valid_every_epochs == 0:
            print('---- validate a child model ----')
            accuracy = final_child.eval(best_arch, valid_imgs, valid_labels)
            final_child_valid_acc.append(accuracy)
            print('epoch: %d, accuracy: %f' % (epoch, accuracy))
            child_file.write('---- validate a child model ----\n') 
            child_file.write('epoch: %d, accuracy: %f\n' % (epoch, accuracy))
    t = time.time() - t
    print('final child training time %.2f sec' % t)
    child_file.write('final child training time %.2f sec \n' % t)
    # ===================
    # test final child
    # ===================
    accuracy = final_child.eval(best_arch, test_imgs, test_labels)
    print('---- test final child ----') 
    print('epoch: %d, accuracy: %f' % (epoch, accuracy))
    child_file.write('---- test final child ----\n') 
    child_file.write('epoch: %d, accuracy: %f\n' % (epoch, accuracy))
    # ===================
    # save final child
    # ===================
    torch.save(final_child.net.graph.state_dict(), final_child_save_path)
    # ===================
    # plot final child training
    # ===================
    plot_metric(final_child_valid_acc, 'final_child_valid_acc', 'final_child_valid_acc')
    
    # how to load models: create then load params
    # example
    # net = Net() # create it
    # net.load_state_dict(torch.load(PATH)) # load parameters of the model
    # close output files    
    child_file.close()
    ctrl_file.close()
    final_child_file.close()

def plot_metric(data, title, file_name):
    """
    Plot the metrics
    """
    x = np.arange(len(data))
    fig = plt.figure(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.bar(x, data)
    plt.title(title, fontsize=15)
    plt.ylabel(title, fontsize=15)
    plt.yticks(fontsize=15)
    fig.savefig(file_name+'.png')

def plot_stack_bar(data, title, file_name):
    """
    Plot the metrics
    """
    x = np.arange(len(data))
    data = np.array(data)
    op_conv3 = data[:, 0]
    op_conv5 = data[:, 1]
    op_avgpool3 = data[:, 2]
    op_maxpool3 = data[:, 3]
    width = 0.4
    fig, ax = plt.subplots()
    ax.bar(x, op_conv3, width, label='conv3')
    ax.bar(x, op_conv5, width, bottom=op_conv3, label='conv5')
    ax.bar(x, op_avgpool3, width, bottom=op_conv3+op_conv5, label='avgpool3')
    ax.bar(x, op_maxpool3, width, bottom=op_conv3+op_conv5+op_avgpool3, label='maxpool3')
    plt.title(title, fontsize=15)
    plt.ylabel(title, fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend()
    fig.savefig(file_name+'.png')    

# ------------------
# Testbench
# ------------------
if __name__ == '__main__':
    main()