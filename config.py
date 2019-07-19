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

# threshold to determine if two images are related
threshold = 0.55 

batch_size = 32

steps_per_epoch = 32
epochs = 256
learning_rate = 0.0001 

l2_regularization = [0, 0.2]
dropout_rate = 0.3
use_dropout = [True, False]

# units in the embedding layers
units_embeddings = [256, 128]
activations = ['relu', 'linear']


# image shape
img_shape = (224, 224, 3)


