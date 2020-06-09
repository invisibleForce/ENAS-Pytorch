"""
child
spec:
    Class controller model is implemented.
    It consists of the model of controller
func list:
    __init__
    build_net - build the nets consisting all the needed layers
    op_sample - sample an operation
    skip_sample - sample skips
    net_sample - sample the whole net
"""
# packages
# std
import os
import sys
# installed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

DEBUG = 0

class StackLSTM(nn.Module):
    """
    StackLSTM class.
    It describes a stacked LSTM which only 
    run a single step.
    """
    def __init__(self, input_size, hidden_size, lstm_num_layers=2):
        # init
        super(StackLSTM, self).__init__() # init the parent class of Net, i.e., nn.Module
        # params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_num_layers = lstm_num_layers
        # create a stacked lstm
        self.net = self._build_net()

    def _build_net(self):
        """
        Create a stacked lstm
        """
        net = []
        # create each layer of the model
        for _ in range(self.lstm_num_layers):
            layer = nn.LSTMCell(self.input_size, self.hidden_size)
            net.append(layer)
        # build it as a nn.ModuleList
        net = nn.ModuleList(net)

        return net

    def __call__(self, inputs, prev_h, prev_c):
        """
        Forward of stacked LSTM
        Args:
            inputs: input of the stack lstm, [batch=1, input_size]
            prev_h & prev_c: hidden and cell states of each layer at the previous time step.
                size = [lstm_num_layer, hidden_size]
        Returns:
            next_h & next_c
        """
        net = self.net
        next_h, next_c = [], []
        for i in range(self.lstm_num_layers):
            if i == 0: x = inputs
            else: x = next_h[-1]
            cur_h, cur_c = net[i](x, (prev_h[i], prev_c[i]))
            next_h.append(cur_h)
            next_c.append(cur_c)
        
        return next_h, next_c

class ControllerModel(nn.Module):
    """
    ControllerModel class.
    It describes the controller model
        
    """
    def __init__(self,
               child_num_layers=6,
               lstm_size=32,
               lstm_num_layers=2,
               num_op=4,
               temperature=5,
               tanh_constant=2.5,
               skip_target=0.4,
               device='gpu'
              ):
        """
        1. init params
        2. create a graph which contains the sampled subgraph
        """
        super(ControllerModel, self).__init__() # init the parent class of Net, i.e., nn.Module
        # parameters for building a child model
        self.child_num_layers = child_num_layers # 
        # parameters for building a controller
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.num_op = num_op
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.skip_target = skip_target
        # device
        self.device = device
        # build controller = net
        # claim all the layers and parameters
        self.net = self._build_net()
        # add g_emb as a parameter to ControllerModel
        # initialized by uniform distribution between -0.1 to 0.1
        # 0 <= torch.rand < 1
        g_emb_init = 0.2 * torch.rand(1,self.lstm_size) - 0.1
        self.register_parameter(name='g_emb', param=torch.nn.Parameter(g_emb_init))
        # results of net sample
        self.sample_arch = []
        self.sample_entropy = []
        self.sample_log_prob = []
        self.sample_skip_count = []
        self.sample_skip_penaltys = []
        

    def _build_net(self):
        """
        Create all the layers with weights
        It is a dict consisting of 'op_sample' and 'skip_sample'.
        op_sample is a dict with 
        """
        net = {}
        # layers & params shared by op and skip
        net['lstm'] = StackLSTM(self.lstm_size, self.lstm_size, self.lstm_num_layers)
        if DEBUG: 
            param = list(net['lstm'].parameters())
            print('lstm')
            for p in param:
                print(p.size())
        # layers & params used only by op
        net['op_fc'] = nn.Linear(self.lstm_size, self.num_op)
        if DEBUG: 
            param = list(net['op_fc'].parameters())
            print('op_fc')
            for p in param:
                print(p.size())
        net['op_emb_lookup'] = nn.Embedding(self.num_op, self.lstm_size)
        if DEBUG: 
            param = list(net['op_emb_lookup'].parameters())
            print('op_emb_lookup')
            for p in param:
                print(p.size())
        # layers & params used only by skip
        net['skip_attn1'] = nn.Linear(self.lstm_size, self.lstm_size) # w_attn1 
        if DEBUG: 
            param = list(net['skip_attn1'].parameters())
            print('skip_attn1')
            for p in param:
                print(p.size())
        net['skip_attn2'] = nn.Linear(self.lstm_size, self.lstm_size) # w_attn2
        if DEBUG: 
            param = list(net['skip_attn2'].parameters())
            print('skip_attn2')
            for p in param:
                print(p.size())
        net['skip_attn3'] = nn.Linear(self.lstm_size, 1)              # v_attn
        if DEBUG: 
            param = list(net['skip_attn3'].parameters())
            print('skip_attn3')
            for p in param:
                print(p.size()) 
        # create a nn.ModuleDict consisting of all the layers used to sample operations
        net = nn.ModuleDict(net)
        
        return net

    def _op_sample(self, args):
        """
        sample an op (it is a part of controller's forward)
        Args: consisting of the following parts
            inputs: input of op_sample
            prev_h & prev_c: the hidden and cell states of the prev layer
            arc_seq: architecture sequence
            log_probs: all the log probabilities used for training (recall the gradient calculation of REINFORCE)
            entropys: all the entropys used for training
        Return:
            x: output of the child model
        """
        net = self.net
        inputs, prev_h, prev_c, arc_seq, log_probs, entropys = args
        # lstm - process hidden states
        next_h, next_c = net['lstm'](inputs, prev_h, prev_c)
        prev_h, prev_c = next_h, next_c
        # fc - calculate logit
        logit = net['op_fc'](next_h[-1])    # h state of the last layer
        # temperature
        if self.temperature is not None:
            logit /= self.temperature
        # tanh and then scaled by a constant
        if self.tanh_constant is not None:
            logit = self.tanh_constant * torch.tanh(logit)
        # use softmax transfer logits to probs
        # or the logits may be negative it can not represent a prob
        prob = f.softmax(logit, dim=1)
        # multinomial for sampling an op
        op_id = torch.multinomial(prob, 1) # logit = probs of each type of operation, 1 = sample a single op
        op_id = op_id[0]
        # generate input for skip_sample using embedding lookup
        inputs = net['op_emb_lookup'](op_id.long())
        # calculate log_prob
        log_prob = f.cross_entropy(logit, op_id)
        # cal entropy
        entropy = log_prob * torch.exp(-log_prob)
        # add op to arc_seq
        if self.device == 'gpu':
            op = op_id.cpu()
        op = int(op.data.numpy()) # to an int
        op = [op] # to list
        arc_seq.append(op)
        # add to log_probs
        log_probs.append(log_prob)
        # add to entropys
        entropys.append(entropy)

        return inputs, prev_h, prev_c, arc_seq, log_probs, entropys        

    def _skip_sample(self, args):
        """
        sample skip connections for layer_id (it is a part of controller's forward)
        Args:
            layer_id: layer count
            inputs: input of op_sample
            prev_h & prev_c: the hidden and cell states of the prev layer
            arc_seq: architecture sequence
            log_probs: all the log probabilities used for training (recall the gradient calculation of REINFORCE)
            entropys: all the entropys used for training
            archors & anchors_w_1: archor points and its weighed values
            skip_targets & skip_penaltys & skip_count: used to enforce the sparsity of skip connections
        Return:
            all args except layer_id
        """    
        layer_id, inputs, prev_h, prev_c, arc_seq, log_probs, entropys, anchors, anchors_w_1, skip_targets, skip_penaltys, skip_count = args
        net = self.net
        # lstm - process hidden states
        next_h, next_c = net['lstm'](inputs, prev_h, prev_c)
        prev_h, prev_c = next_h, next_c
        if layer_id > 0:
            # use attention mechanism to generate logits
            # concate the weighed anchors
            query = torch.cat(anchors_w_1, dim=0) 
            # attention 2 - fc
            query = torch.tanh(net['skip_attn2'](next_h[-1]) + query)
            # attention 3 - fc            
            query = net['skip_attn3'](query)
            # generate logit
            logit = torch.cat([-query, query], dim=1)
            # process logit with temperature
            if self.temperature is not None:
                logit /= self.temperature
            # process logit with tanh and scale it
            if self.temperature is not None:
                logit = self.tanh_constant * torch.tanh(logit)
            # calculate prob of skip (see NAS paper, Sec3.3)
            skip_prob = torch.sigmoid(logit) # use sigmoid to convert skip to its prob
            # sample skip connections using multinomial distribution sampler
            skip = torch.multinomial(skip_prob, 1)  # 0 - used as an input, 1 - not an input
            # calcualte kl as skip penalty
            kl = skip_prob * torch.log(skip_prob / skip_targets) # calculate kl
            kl = torch.sum(kl)
            skip_penaltys.append(kl)
            # cal log_prob and append it - used by REINFORCE to calculate gradients of controller (i.e., LSTM)
            log_prob = f.cross_entropy(logit, skip.squeeze(dim=1))
            log_probs.append(torch.sum(log_prob))
            # cal entropys and append it
            entropy = log_prob * torch.exp(-log_prob)
            entropy = torch.sum(entropy)
            entropys.append(entropy)
            # update count of skips
            skip_count.append(skip.sum())
            # add skip to arc_seq
            if self.device == 'gpu':
                skip_cpu = skip.cpu()
            arc_seq.append(skip_cpu.squeeze(dim=1).data.numpy().tolist())
            # generate inputs for the next time step
            skip = torch.reshape(skip, (1, layer_id)) # reshape skip
            cat_anchors = torch.cat(anchors, dim=0)
            # skip = 1 x layer_id (layer_id > 0) 
            # cat_anchors = layer_id x lstm_size
            inputs = torch.matmul(skip.float(), cat_anchors) 
            inputs /= (1.0 + torch.sum(skip))
        else:
            inputs = self.g_emb
            if self.device == 'gpu':
                inputs = inputs.cuda()
            arc_seq.append([]) # no skip, use empty list to occupy the position
        
        # cal the
        anchors.append(next_h[-1])
        # cal attention 1
        attn1 = net['skip_attn1'](next_h[-1])
        anchors_w_1.append(attn1)

        return inputs, prev_h, prev_c, arc_seq, log_probs, entropys, anchors, anchors_w_1, skip_targets, skip_penaltys, skip_count

    def net_sample(self):
        """
        run (like forward) a controller model to sample an neural architecture
        Args:
            
        Return:
            
        """
        # net sample
        arc_seq = []
        entropys = []
        log_probs = []
        # skip sample 
        anchors = []        # store hidden states of skip lstm; anchor = hidden states of skip lstm (i.e., layer_id)
        anchors_w_1 = []    # store results of attention 1 (input=h, w_attn1)
        skip_count = []
        skip_penaltys = []

        # determine the device used to run the model
        if self.device == 'gpu': # check whether gpu is available or not
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else: device = 'cpu'
        if DEBUG: print(device)
        # move model to gpu
        if self.device == 'gpu': # check whether gpu is available or not
            self.net.to(device) # move net to gpu
        # init inputs and states
        # init prev cell states to zeros for each layer of the lstm
        prev_c = [torch.zeros((1, self.lstm_size),device=device) for _ in range(self.lstm_num_layers)]
        # init prev hidden states to zeros for each layer of the lstm
        prev_h = [torch.zeros((1, self.lstm_size),device=device) for _ in range(self.lstm_num_layers)]
        # inputs
        inputs = self.g_emb
        if self.device == 'gpu': # check whether gpu is available or not
            inputs = inputs.cuda()
        # skip_target = 0.4 = the prob of a layer used as an input of another layer
        # 1 - skip_target = 0.6; the probability that this layer is not used as an input
        skip_targets = torch.tensor([1.0 - self.skip_target, self.skip_target], dtype=torch.float, device=device)
        

        # sample an arch
        for layer_id in range(self.child_num_layers):
            arg_op_sample = [inputs, prev_h, prev_c, arc_seq, log_probs, entropys]
            returns_op_sample = self._op_sample(arg_op_sample)
            inputs, prev_h, prev_c, arc_seq, log_probs, entropys = returns_op_sample
            arg_skip_sample = [layer_id, inputs, prev_h, prev_c, arc_seq, log_probs, entropys, 
                                anchors, anchors_w_1, skip_targets, skip_penaltys, skip_count]
            returns_skip_sample = self._skip_sample(arg_skip_sample)
            inputs, prev_h, prev_c, arc_seq, log_probs, entropys, anchors, anchors_w_1, skip_targets, skip_penaltys, skip_count = returns_skip_sample

        # generate sample arch
        # [[op], [skip]] * num_layer
        self.sample_arch = arc_seq
        if DEBUG: 
            print('sample_arch')
            print('len:', len(self.sample_arch))
            for idx, data in enumerate(self.sample_arch):
                if idx % 2 == 0:
                    print('-' * 15)
                    print('layer:', idx)
                    print('op:', data)
                else:
                    print('skip:', data)
        # cal sample entropy
        entropys = torch.stack(entropys)
        self.sample_entropy = torch.sum(entropys)
        if DEBUG: 
            print('sample_entropy: %.3f' % self.sample_entropy.item())
            
        # cal sample log_probs
        log_probs = torch.stack(log_probs)
        self.sample_log_prob = torch.sum(log_probs)
        if DEBUG: 
            print('sample_log_prob: %.3f' % self.sample_log_prob.item())
            
        # cal skip count
        skip_count = torch.stack(skip_count)
        self.sample_skip_count = torch.sum(skip_count)
        if DEBUG: 
            print('sample_skip_count: %.0f' % self.sample_skip_count.item())
            
        # cal skip penaltys
        skip_penaltys = torch.stack(skip_penaltys)
        self.sample_skip_penaltys = torch.sum(skip_penaltys)
        if DEBUG: 
            print('sample_skip_penaltys : %.3f' % self.sample_skip_penaltys.item())
        

def test_model():
    ctrler = ControllerModel()
    # param = list(ctrler.parameters())
    # param_num = len(param)
    # print(param_num)
    # for i in range(param_num):
    #     # print(p.size())
    #     print(i, param[i].size())
    print(ctrler.net)
    ctrler.net_sample()
    
# ------------------
# Testbench
# ------------------
if __name__ == '__main__':
    test_model()