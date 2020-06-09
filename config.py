class enas_cfg():
    # -----------
    # platform 
    # -----------
    device = 'gpu'

    # -----------
    # enas 
    # -----------
    epoch_num = 4
    retrain_epoch_num = 1
    # -----------
    # child 
    # -----------
    # model
    child_class_num = 10
    child_num_layers = 6
    child_out_channels = 32
    child_num_op = 4
    # --- training
    child_data_path = 'F:/2 Work/0 Solo/data/cifar-10-python/'
    child_num_valids = 5000
    child_batch_size = 32
    child_opt_algo = 'sgd' # not used indeed
    # learning scheduler = cosine anealing
    child_lr_init = 0.05
    child_lr_gamma = 0.1
    child_lr_cos_lmin = 0.001
    child_lr_cos_Tmax = 2
    # weight decay = l2 regularization
    child_l2_reg = 1e-4
    # optimizer = sgd + Nestrov momentum
    # log
    child_run_loss_every = 100
    # validate shared parameters
    child_valid_every_epochs = 1
    

    # -----------
    # controller 
    # -----------
    # model
    ctrl_lstm_size = 32
    ctrl_lstm_num_layers = 2
    # child
    # --- training
    ctrl_train_step_num = 10 # number of training steps per epoch
    ctrl_batch_size = 5 # number of samples per training step
    ctrl_opt_algo = 'adam' # not used indeed
    ctrl_train_every_epochs = 2
    # learning scheduler = exponential decaying
    ctrl_lr_init = 0.0001
    ctrl_lr_gamma = 0.1
    # baseline - reduce high variance; exponential moving average
    ctrl_baseline_decay = 0.999
    # prevent from being permature of controller
    # applied to logits
    ctrl_temperature = 5
    ctrl_tanh_constant = 2.5
    # add entropy to reward
    ctrl_entropy_weight = 0.0001
    # enforce skip sparsity 
    # add skip penalty to loss
    ctrl_skip_target = 0.4
    ctrl_skip_weight = 0.8
    # validate/test controller 
    ctrl_valid_every_epochs = 1
    ctrl_eval_arc_num = 2
    ctrl_final_arc_num = 2
    # -----------
    # output 
    # -----------
    child_filename = 'log_child.txt'
    ctrl_filename = 'log_controller.txt'
    final_child_filename = 'log_final_child.txt'
    child_model_save_path = './enas_child.pth' # save params which can be reused later. may not need to trian the child again.
    ctrl_model_save_path = './enas_ctrl.pth' # save params which can be reused later. may not need to trian the child again.
    final_child_save_path = './final_child.pth' # save params which can be reused later. may not need to trian the child again.