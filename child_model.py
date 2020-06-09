"""
child
spec:
    Class ChildModel is implemented.
func list:
    __init__
    build_graph - greate a large graph
    model - forward of a subgraph
        nas_layer
"""
# packages
# std
import os
import sys
# installed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim # optimizer
# home made
from child_layer import NasLayer
from child_layer import global_avgpool

DEBUG = 0

class ChildModel(nn.Module):
    """
    Child class.
    It describes functions for 
        1) building a model: it is the same as a forward of a net
        def model
        2) training a child
        def build_train:
        3) validating a child
        def build_valid
        4) testing a child
        def build_test
        5) validating a child for RL
        def build_valid_rl
    """
    def __init__(self,
               class_num,
               num_layers=6,
               out_channels=24,
               batch_size=32
              ):
        """
        1. init params
        2. create a graph which contains the sampled subgraph
        """
        super(ChildModel, self).__init__() # init the parent class of Net, i.e., nn.Module
        # data set used for training, validating, testing
        self.class_num = class_num # number of classes
        # parameters for building a child model
        self.num_layers = num_layers # 
        self.out_channels = out_channels
        # build DAG = net
        self.graph = self.build_graph(self.class_num)
        

    def build_graph(self, class_num):
        """
        Create a large graph which contains the sampled subgraph
        stem_conv: input [N, C=3, H=24, W=24]; kernel [M, C=3, R=3, P=3]/[Sx=1, Sy=1]
        nas layers
        global avg pool
        fc
        """
        graph = []
        # stem_conv: process the input image
        # add padding zeros to let ifmap and ofmap have the same size
        px = int((3 - 1) / 2)
        py = px
        padding_size = (px, py)
        stem_conv = nn.Conv2d(
                in_channels=3, 
                out_channels=self.out_channels, 
                kernel_size = 3, 
                padding=padding_size,
                stride=1)
        graph.append(stem_conv)
        # major part of the graph consisting of all NasLayers
        for _ in range(self.num_layers):
            graph.append(NasLayer(self.out_channels))
        # fc
        fc = nn.Linear(self.out_channels, class_num, bias=True)
        graph.append(fc)
        # create a ModuleList, or the parameters cannot be added
        graph = nn.ModuleList(graph)

        return graph

    def model(self, x, sample_arch):
        """
        run (like forward) a child model determined by sample_arch
        Args:
            sample_arch: a list consisting of 2 * num_layers elements
                op_id = sample_arch[2k]: operation id
                skip = sample_arch[2k + 1]: element i of such abinary vector 
                    is used to describe whether the previous layer i is used 
                    as an input
            x: input of the child model
        Return:
            x: output of the child model
        """
        # layers
        prev_layers = []
        # stem_conv
        x = self.graph[0](x)
        prev_layers.append(x)
        offset = 1
        # nas_layers
        for cnt_layer in range(self.num_layers):
            layer_config = sample_arch[2 * cnt_layer : 2 * cnt_layer + 2]   # [op], [skip]
            x = self.graph[cnt_layer + offset](cnt_layer, prev_layers, layer_config)
            prev_layers.append(x)
        # global_avgpool
        x = global_avgpool(x)
        # fc
        x = self.graph[-1](x)

        return x


def test_model():
    # child model arch
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
    # instantiate a model
    images = torch.rand([2, 3, 7, 7])
    labels = torch.tensor([1, 2])
    class_num = 5
    num_layers = 6
    out_channels = 2
    child = ChildModel(class_num, num_layers, out_channels)
    print(len(list(child.parameters())))
    # print(list(child.parameters()))
    print(len(child.graph))
    print(child.graph)
    y = child.model(images, sample_arch)
    print(y.size())

# ------------------
# Testbench
# ------------------
if __name__ == '__main__':
    test_model()