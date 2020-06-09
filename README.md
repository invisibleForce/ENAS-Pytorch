# ENAS-Pytorch
A pytorch implementation of ENAS
It only implements the macro search for CIFAR-10 dataset
(i.e., search the operation and input connections of the whole CNN).
The operation of each layer can be one of conv3, conv5, avgpool3, maxpool3 with stride 1.

# Main procedure of this implemented ENAS
Phase 1: Train child (shared params) and controller
  1. sample a child (i.e., sampled arch) including operation and its inputs of each layer
  2. train the sampled child 
  3. validate the child if condition satisfied
  4. train the controller if condition satisfied
  5. validate the controller if condition satisfied
  
Phase 2: Derive the best arch 
  1. derive an arch after controller samples 1000 archs. 
  The arch obtains the best accuracy on validation dataset is selected, referred to as best arch.
  2. retrain the best arch from scratch. 

# Ref
Efficient Neural Architecture Search via Parameter Sharing
Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, Jeff Dean
https://arxiv.org/abs/1802.03268

# Best result
A 14-layer graph is searched and its accuracy on the test dataset is 84% after retraining.

# Contact
Hope this project is helpful to you.
I am new to learn RL and am very glad to discuss with anyone. 
Feel free to comment or send me email.
email: invisibleforce@163.com
