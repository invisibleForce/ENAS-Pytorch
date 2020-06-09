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
import pickle 
import torch

DEBUG = 0

def unpickle(file):
    """
    Read a batch
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def _read_data(file_list):
    """
    Read a dataset and reshape it
    args: 
        file_list: path of dataset
    returns:
        data_set: all data
            format NCHW: x, 3072 -> x, 3, 32, 32
        label_set: all labels of the data
    """
    for i in range(len(file_list)):
        file = file_list[i]
        if DEBUG: print(file)
        data_batch = unpickle(file)
        data = data_batch[b'data']
        data = data.astype('float32')   # change to 
        data = data.reshape((-1, 3, 32, 32))
        labels = np.array(data_batch[b'labels']).astype('int')
        if i == 0:
            data_set = data
            label_set = labels
        else:
            data_set = np.concatenate((data_set, data), axis = 0)
            label_set = np.concatenate((label_set, labels), axis = 0)
        # if DEBUG:
        #     print(data.shape)
        #     print(labels.shape)


    return (data_set, label_set)

def read_data(data_path, num_valids=5000):
    """
    Read train/valid/test data sets
    args: 
        num_valids: num of images in a valid set
    returns:
        images: N, C, H, W
            in case of CIFAR-10, it consists of 
            
            train: 
                no valid set: 50k
                with valid set: 45k
            test:
                10k
            valid
                last 5k of training set
        label_set: all labels of the datasets
    """
    # create the file list
    # data_path = 'F:/2 Work/0 Solo/data/cifar-10-python/'
    train_files = []
    for i in range(5):
        # train_files.append('../data/cifar-10-python/data_batch_' + str(i+1))
        train_files.append(data_path + 'data_batch_' + str(i+1))
        # path:
            # F:\2 Work\0 Solo invented by InvisibleForce\pytorch\data\cifar-10-batches-py
    test_files = []
    test_files.append(data_path + 'test_batch')

    # create image and label dict
    images, labels = {}, {}
    # read train set
    images['train'], labels['train'] = _read_data(train_files)
    # read valid set
    if num_valids: # need
        # valid set
        images['valid'] = images['train'][-num_valids:] # last num_valids images
        labels['valid'] = labels['train'][-num_valids:]
        # train set = the remaining orginal train set
        images['train'] = images['train'][:-num_valids] # last num_valids images
        labels['train'] = labels['train'][:-num_valids]

    # read test set
    images['test'], labels['test'] = _read_data(test_files)


    # normalize data
    # 1. sub mean
    # 2. divide std
    # proc train set
    images['train'] = normalize(images['train'])
    # convert data to tensor
    images['train'] = torch.from_numpy(images['train'])
    # proc valid set
    if num_valids:
        images['valid'] = normalize(images['valid'])
        images['valid'] = torch.from_numpy(images['valid'])
    # proc test set
    images['test'] = normalize(images['test'])
    images['test'] = torch.from_numpy(images['test'])
    # convert labels from np array to torch.tensor.long
    labels['train'] = torch.from_numpy(labels['train'])
    labels['train'] = labels['train'].long()
    if num_valids:
        labels['valid'] = torch.from_numpy(labels['valid'])
        labels['valid'] = labels['valid'].long()
    labels['test'] = torch.from_numpy(labels['test'])
    labels['test'] = labels['test'].long()
    if DEBUG: print('read_data, images.train', type(images['train']))
    if DEBUG: print('read_data, images.valid', type(images['valid']))
    if DEBUG: print('read_data, images.test', type(images['test']))
    if DEBUG: print('read_data, labels.train', type(labels['train']))
    if DEBUG: print('read_data, labels.valid', type(labels['valid']))
    if DEBUG: print('read_data, labels.test', type(labels['test']))

    return images, labels

def normalize(dataset):
    """
    1. sub mean
    2. divide std
    arg:
        img: dataset, (N, C, H, W)
    return: 
        img: normalized dataset
    """

    dataset = np.transpose(dataset, [0, 2, 3, 1]) # NCHW -> NHWC
    dataset = dataset / 255.0 # 0-255 -> 0-1
    mean = np.mean(dataset, axis=(0, 1, 2), keepdims=True)
    std = np.std(dataset, axis=(0, 1, 2), keepdims=True)
    dataset = (dataset - mean) / std 
    dataset = np.transpose(dataset, [0, 3, 1, 2]) # NHWC -> NCHW

    return dataset

def augment(batch):
    """
    Processed on GPU
    1 upsample: 32x32 -> 40x40
    2 randomly crop: 40x40 -> 32x32
    3 flip horizontally: left -> right
    arg:
        img: a batch of images, (N, C, H, W)
    return: 
        img: augmented batch
    """
    # convert batch from tensor to nparray
    # batch = batch.data.numpy() # only supported by cpu devices
    if DEBUG: print('augment', type(batch))
    # parameters
    N, C, H, W = batch.size()
    # augment
    # for i in range(N):
    # img = batch[i, :, :, :]
    # 1 upsample: 32x32 -> 40x40 (H, W)
    batch = pad(batch, [[4, 4], [4, 4]])
    # 2 randomly crop: 40x40 -> 32x32
    batch = crop(batch, [H, W])
    # 3 flip horizontally: left -> right
    batch = flip_left_right(batch)
    # store it back to batch
    # batch[i, :, :, :] = img_flip

    # batch = torch.from_numpy(batch) # only for cpu
    if DEBUG: print('augment', type(batch))
    return batch 

def pad(batch, pad_size):
    """
    pad zeros to an img
    arg:
        img: an image, (C, H, W)
        pad_size: [[C_before, C_after], [H_top, H_bottom], [W_left, W_right]]
    return: 
        img: a zero-padded img
    """
    # params
    N, C, H, W = batch.size()
    H_zeros, W_zeros = pad_size
    # zero padding
    H_zp = H + H_zeros[0] + H_zeros[1]
    W_zp = W + W_zeros[0] + W_zeros[1]
    batch_zp = torch.zeros((N, C, H_zp, W_zp)).cuda()
    # batch_zp = batch_zp
    batch_zp[:, :, H_zeros[0] : H_zeros[0] + H, W_zeros[0] : W_zeros[0] + W] = batch
    
    return batch_zp

def crop(batch, crop_size):
    """
    randomly crop img to a smaller one
    arg:
        img: an image, (C, H, W)
        crop_size: [C_crop, H_crop, W_crop]
    return: 
        img_crop: a cropped img
    """
    # parameter
    N, C, H, W = batch.size()
    H_crop, W_crop = crop_size
    H_diff = H - H_crop + 1
    W_diff = W - W_crop + 1
    # randomly sample the crop offset
    H_offset = int(np.random.randint(0, H_diff, size=1))
    W_offset = int(np.random.randint(0, W_diff, size=1))
    # crop the img
    batch_crop = batch[:, :, H_offset : H_offset + H_crop, W_offset : W_offset + W_crop]

    return batch_crop

def flip_left_right(batch):
    """
    flip img from left to right. 
    CHW data format needs to flip axis=2 (i.e., W-axis) 
    arg:
        img: an image, (C, H, W)
    return: 
        img_flip: a left-right-flipped img 
    """
    # img_flip = np.flip(img, axis=2)
    # batch = torch.flip(batch, )
    batch = batch.flip(3) # flip along w-axis

    return batch



def test_data_proc():
    images, labels = read_data()
    batch = images['train'][0:3]
    print(batch[0, 0, 5, :])
    print(torch.sum(batch))
    batch_aug = augment(batch)
    print(batch_aug[0, 0, 5, :])
    print(torch.sum(batch_aug))
    # diff = batch - batch_aug
    # print(batch)
    # print(batch_aug)
    # print(diff)
    print(batch.shape)
    print(batch_aug.shape)

# ------------------
# Testbench
# ------------------
if __name__ == '__main__':
    test_data_proc()