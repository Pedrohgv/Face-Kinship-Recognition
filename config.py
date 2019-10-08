#!/usr/bin/env python3
# -*- coding = utf-8 -*-
"""
Created on Mon Jul  1 19 =00 =23 2019

@author = pedro
"""

# =============================================================================
# Contains configrations and hyperparameters for training
# =============================================================================

# positive-to-negative distance margin
distance_margin = 0.5

# threshold to determine if two images are related (if using inception resnet V2 model)
threshold = 0.55

batch_size = 32

steps_per_epoch = 32
epochs = 1024
learning_rate = 0.0001

l2_regularization = [0, 0.2]
dropout_rate = 0.3
use_dropout = [True, False]

set_seed_datagen = True
seed_datagen = 1

# units in the embedding layers (if using inception resnet V2 model)
units_embeddings = [256, 128]
activations = ['relu', 'linear']

# training set size
number_train_families = 377

# image shape
img_shape = (160, 160, 3)

"""
Only last 2 layers trained (dense layer and batch normalization)

Variable 'alpha' from L2Norm2Prob layer is trainable initialized with Glorot Uniform, and regularied with l2=0.01
Variable beta added.
"""
