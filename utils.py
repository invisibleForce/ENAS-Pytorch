"""
spec:
    all functions used to process data
func list:
    read_data
        _read_data
            unpickle
        nomalize
    augment
        pad
        crop
        flip_left_right
"""


import numpy as np
import matplotlib.pyplot as plt
import torch

DEBUG = 0

def print_sample_arch(sample_arch, file):
    """
    print a sample arch to a file
    """
    num_layer = len(sample_arch) // 2
    file.write('sample_result\n')
    file.write('layer\t'+'op\t'+'inputs\t\n')
    for i in range(num_layer):
        file.write(str(i)+'\t'+str(sample_arch[2 * i])+'\t'+str(sample_arch[2 * i + 1])+'\n')
    file.write('_' * 20+'\n')
    file.write('arch'+'\n')
    file.write('layer\t'+'op\t'+'inputs\t'+'\n')
    file.write('NOTE: inputs are used by next layer indeed, except layer0'+'\n')
    for i in range(num_layer):
        op = _convert_op(sample_arch[2 * i])
        inputs = _convert_inputs(sample_arch[2 * i + 1])
        file.write(str(i)+'\t'+op+'\t'+str(inputs)+'\n')


def display_sample_arch(sample_arch):
    """
    print a sample arch
    """
    num_layer = len(sample_arch) // 2
    print('sample_result')
    print('layer\t'+'op\t'+'inputs\t')
    for i in range(num_layer):
        print(str(i)+'\t'+str(sample_arch[2 * i])+'\t'+str(sample_arch[2 * i + 1]))
    print('_' * 20)
    print('arch')
    print('layer\t'+'op\t'+'inputs\t')
    print('NOTE: inputs are used by next layer indeed, except layer0')
    for i in range(num_layer):
        op = _convert_op(sample_arch[2 * i])
        inputs = _convert_inputs(sample_arch[2 * i + 1])
        print(str(i)+'\t'+op+'\t'+str(inputs))

def _convert_op(num):
    """
    print a sample arch
    op 
    0 - conv3 - c3
    1 - conv5 - c5
    2 - avgpool3 - ap3
    4 - maxpool3 - mp5
    """
    num = num[0]
    op = []
    op.append('conv3')
    op.append('conv5')
    op.append('avgpool3')
    op.append('maxpool3')
    return op[num]

def _convert_inputs(inputs):
    """
    convert inputs from binary config to indices of inputs
    """
    num_inputs = len(inputs)
    used_inputs = []
    if num_inputs:
        for idx in range(num_inputs):
            if inputs[idx]: 
                used_inputs.append(idx)
        used_inputs.append(num_inputs)
    else: used_inputs.append('img')
    return used_inputs

def test():
    # print_sample_arch
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
    print_sample_arch(sample_arch)
# ------------------
# Testbench
# ------------------
if __name__ == '__main__':
    test()