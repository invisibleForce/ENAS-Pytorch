"""
NasLayer
spec:
    Class NasLayer implements the op and skip connections needed in a layer used in NAS.
        Op: It can be one of conv3, conv5, avgpool3 and maxpool3. Their strides are 1.
            An op is indexed by a number. conv3 = 0, conv5=1, avgpool3=2, maxpool3=3.

        Skip: It combines all the needed previous layers. 
            It is represented by a binary vector consisting of i - 1 elements for Layer i. 
            When the element j is 1, the corresponding layer j (j = [0, i - 1]) is used as inputs.
            Note that skip of the layer i is actually used as the input of layer i + 1.
    Class LayerOp implements an operation needed by a NasLayer. 
        It consists of several fundamental layers such as conv, bn, etc. 
    
func list:
    __init__
    forward

--------------------------
log
--------------------------
5.18-5.19
    1. LayerOp
    2. NasLayer
5.25 
1. use nn.ModuleList
"""
# packages
# std
import os
import sys
# installed
import numpy as np
import torch 
import torch.nn as nn
# home made

DEBUG = 0


def global_avgpool(x):
    """
    An operation used to reduce the H and W axis
    x = [N, C, H, W] -> [N, C, 1, 1]
    """
    H = x.size()[2]
    W = x.size()[3]
    x = torch.sum(x, dim=[2, 3])
    x = x / (H * W)

    return x

class LayerOp(nn.Module):
    """
    An operation used by a nas layer
    Args:
        op: conv3, conv5, avgpool3, maxpool3
        out_channels: = M, num of filters
    Note:
        conv3/5: need to pad zeros to let the input and output 
        feature maps have the same size. The size padded zeros
        are given as follows. 
        ofmap size:
        E = np.floor((H + 2px - 1 * (R - 1) - 1) / Sx + 1) # see pytorch nn.conv2d for details
        F = np.floor((H + 2px - 1 * (R - 1) - 1) / Sx + 1)
        Let E = H, F = W, so we can solve px and py as follows
        Height: px = (R - 1) / 2
        Width: py = (P - 1) / 2
    """
    def __init__(self, op, out_channels):
        # parameters
        self.op = op
        self.out_channels = out_channels
        # init
        super(LayerOp, self).__init__() #
        self.layer_list = self._build_layer()
    
    def _build_layer(self):
        """
        Build a layer consisting of several layers in a list.
        nn.ModuleList is used to register parameters of layers in the list.
        """
        layer_list = []
        # conv_in
        conv_in = nn.Conv2d(
            in_channels=self.out_channels, 
            out_channels=self.out_channels, 
            kernel_size = 1, 
            stride=1)
        layer_list.append(conv_in)
        # bn_in
        bn_in = nn.BatchNorm2d(num_features=self.out_channels)
        layer_list.append(bn_in)
        # relu_in
        relu_in = nn.ReLU()
        layer_list.append(relu_in)
        # kernel
        if self.op == 'conv3':
            # add padding zeros to let ifmap and ofmap have the same size
            px = int((3 - 1) / 2)
            py = px
            padding_size = (px, py)
            kernel = nn.Conv2d(
            in_channels=self.out_channels, 
            out_channels=self.out_channels, 
            kernel_size = 3, 
            padding=padding_size,
            stride=1)
        elif self.op == 'conv5':
            px = int((5 - 1) / 2)
            py = px
            padding_size = (px, py)
            kernel = nn.Conv2d(
                in_channels=self.out_channels, 
                out_channels=self.out_channels, 
                kernel_size = 5, 
                padding=padding_size,
                stride=1)
        elif self.op == 'avgpool3':
            # add padding zeros to let ifmap and ofmap have the same size
            px = int((3 - 1) / 2)
            py = px
            padding_size = (px, py)
            kernel = nn.AvgPool2d(
                kernel_size=3, 
                padding=padding_size,
                stride=1)
        elif self.op == 'maxpool3':
            # add padding zeros to let ifmap and ofmap have the same size
            px = int((3 - 1) / 2)
            py = px
            padding_size = (px, py)
            kernel = nn.MaxPool2d(
                kernel_size=3, 
                padding=padding_size,
                stride=1)
        layer_list.append(kernel)
        # bn_out
        if (self.op == 'conv3') or (self.op == 'conv5'):
            bn_out = nn.BatchNorm2d(num_features=self.out_channels)
            layer_list.append(bn_out)
        # create a ModuleList which will register all the parameters
        layer_list = nn.ModuleList(layer_list)

        return layer_list

    def __call__(self, x):
        """
        Forward of an operation used by a nas layer
        Args:
            x: ifmap
        Return
            x: ofmap
        """
        for layer in self.layer_list:
            x = layer(x)

        return x
        

class NasLayer(nn.Module):
    """
    NasLayer
    """
    def __init__(self, out_channels=24):
        """
        Create a nas layer
        """
        # parameters
        self.out_channels = out_channels
        # ops
        super(NasLayer, self).__init__() #init
        self.layer_list = self._build_nas_layer()
        

    def _build_nas_layer(self):
        """
        build a nas layer consisting all possible branches
        """
        layer_list = []
        # conv3, 0
        conv3 = LayerOp('conv3', self.out_channels)
        layer_list.append(conv3)
        # conv5, 1
        conv5 = LayerOp('conv5', self.out_channels)
        layer_list.append(conv5)
        # avgpool3, 2
        avgpool3 = LayerOp('avgpool3', self.out_channels)
        layer_list.append(avgpool3)
        # maxpool3, 3
        maxpool3 = LayerOp('maxpool3', self.out_channels)
        layer_list.append(maxpool3)
        # bn_out
        bn_out = nn.BatchNorm2d(num_features=self.out_channels)
        layer_list.append(bn_out)
        # create a module list
        layer_list = nn.ModuleList(layer_list)
    
        return layer_list

    def layer_op(self, x, op):
        """
        Run the operation of a nas layer
        Args:
            x: ifmap
            op: operation to run
                0 - conv3
                1 - conv5
                2 - avgpool3
                3 - maxpool3
        Returns:
            x: ofmap
        """
        x = self.layer_list[op[0]](x)
        if DEBUG: print('op', op)
        
        return x
        
    def skip(self, prev_layers, config):
        """
        Comcate the desired preve layers of a nas layer
        Args:
            prev_layers: previous layers
            config: describe all the combined layers
        Returns:
            y: ofmap
        """
        # add all the desired prev layers together
        offset = 1 # used to skip the stem_conv
        num_layer = len(prev_layers) - offset
        # x = torch.zeros(prev_layers[0].size()) 
        x = []
        for i in range(num_layer):
            if config[i]:
                x.append(prev_layers[i + offset])
                # if DEBUG: print('input=',i)
        if len(x):
            x = torch.stack(x) # stack all the tensors in an additional axis (i.e., 0)
            if DEBUG: print(x.size())
            x = torch.sum(x, dim=0) # add along axis 0
            if DEBUG: print(x.size())
        else:
            x = torch.zeros(prev_layers[0].size()) 
            x = x.cuda()
        return x


    def __call__(self, cnt_layer, prev_layers, layer_config):
        """
        describe the forward of the layer
        Args:
            prev_layers: all previous layers
            layer_config: op and connectivity
        """
        if DEBUG: print('layer', cnt_layer)
        # input
        x = prev_layers[-1]
        # if DEBUG: print('input\n', x.data)
        # run op of the enas layer
        op_config = layer_config[0]
        x = self.layer_op(x, op_config)
        # if DEBUG: print('op_out\n', x.data)
        if cnt_layer > 0:
            # combine the skip (add skips with x)
            skip_config = layer_config[1]
            y = self.skip(prev_layers, skip_config)
            # if DEBUG: print('skip_out\n', y.data)
            # combine op and skip results
            # gpu not supporting x + y
            x = [x, y]
            x = torch.stack(x)
            if DEBUG: print(x.size())
            x = torch.sum(x, dim=0)
            if DEBUG: print(x.size())
            # x = x + y
            # if DEBUG: print('op+skip\n', x.data)
            x = self.layer_list[-1](x) # bn_out
            # if DEBUG: print('final_out\n', x.data)

        return x
        


# ------------------
# Test functions
# ------------------
def test_layer_op():
    N = 1
    C = 2
    H = 7
    W = 7
    M = 2
    img_size = (N, C, H, W)
    img = torch.rand(img_size)
    op = 'conv3'
    # op = 'conv5'
    # op = 'avgpool3'
    # op = 'maxpool3'
    branch = LayerOp(op, M)
    print(len(list(branch.parameters())))
    print(list(branch.parameters()))
    y = branch(img)
    
    print(y.size())
    print(y.data)

def test_nas_layer():
    # prev_layers
    N = 1
    C = 4
    H = 7
    W = 7
    M = C
    layer_size = (N, C, H, W)
    layer_num = 4
    prev_layers = []
    for i in range(layer_num):
        prev_layers.append(torch.rand(layer_size))
    # layer_config
    op = 0  # conv3
    op = 1  # conv5
    # op = 2  # avgpool3
    # op = 3  # maxpool3
    skip = [0, 0, 1, 1]
    layer_config = [[op], skip]
    layer = NasLayer(M)
    print(len(list(layer.parameters())))
    print(list(layer.parameters()))
    # run
    cnt_layer = 4
    y = layer(cnt_layer, prev_layers, layer_config)
    print(y.size())
    print(y.data)
# ------------------
# Testbench
# ------------------
if __name__ == '__main__':
    # test_layer_op()
    test_nas_layer()