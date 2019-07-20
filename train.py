# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:07:20 2019

@author: btq9
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import os

# import some hyperparameters
from config import img_shape

from config import batch_size
from config import steps_per_epoch
from config import epochs
from config import learning_rate

from config import units_embeddings
from config import activations
from config import use_dropout
from config import dropout_rate
from config import l2_regularization

# import some functions
from input_manager import process_data, make_triplet_dataset
from training_log_callbacks import CallbackPlot, CallbackSaveLogs, create_folder
from losses import triplet_loss, accuracy, pos_dist, neg_dist
from models import create_fully_connected_layers, make_inception_resnet_v2_model

# imports pre-trained Inception ResNet V2 model
from tensorflow.keras.applications import InceptionResNetV2

# imports from keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate


def initialize_devices():
    '''
    Initializes GPU devices. Memory growth must be set for GPU if a GPU is to be used.
    This was necessary with this build of tensorflow, GPU-tensorflow-beta.
    THIS MUST BE RUN AS THE FIRST FUNCTION IN ORDER FOR TENSORFLOW TO BE ABLE TO USE A GPU
    '''
    # Enables device placement logging to show the devices operations are using to do computaions
    # tf.debugging.set_log_device_placement(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPU's
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPU's have been initialized
            print(e)

    return None


def train(save_model=False):

    train_families, test_families, train_positive_relations, test_positive_relations = process_data(
        set_seed=True)

    # constructs training data pipeline. lambda funtion with parameters is passed instead of a simple function and parameters
    # being passed as arguments separately because the parameters are not tensors.
    train_dataset = tf.data.Dataset.from_generator(lambda: make_triplet_dataset(train_families, train_positive_relations),
                                                   output_types=({'input_anchor': tf.float32, 'input_positive': tf.float32, 'input_negative': tf.float32}, tf.int32))
    train_dataset = train_dataset.batch(batch_size)

    # constructs test data pipeline
    test_dataset = tf.data.Dataset.from_generator(lambda: make_triplet_dataset(test_families, test_positive_relations),
                                                  output_types=({'input_anchor': tf.float32, 'input_positive': tf.float32, 'input_negative': tf.float32}, tf.int32))
    test_dataset = test_dataset.batch(batch_size)

    loss_plot_settings = {'variables': {'loss': 'Training loss',
                                        'val_loss': 'Validation loss'},
                          'title': 'Losses',
                          'ylabel': 'Epoch Loss',
                          'last_50': False}

    last_50_plot_settings = {'variables': {'loss': 'Training loss',
                                           'val_loss': 'Validation loss'},
                             'title': 'Losses from last 50 epochs',
                             'ylabel': 'Epoch Loss',
                             'last_50': True}

    accuracy_plot_settings = {'variables': {'accuracy': 'Training accuracy',
                                            'val_accuracy': 'Validation accuracy'},
                              'title': 'Accuracies from last 50 epochs',
                              'ylabel': 'Epoch Accuracy',
                              'last_50': True}

    distances_plot_settings = {'variables': {'pos_dist': 'Training positive distances',
                                             'neg_dist': 'Training negative distances',
                                             'val_pos_dist': 'Validation positive distances',
                                             'val_neg_dist': 'Validation negative distances'},
                               'title': 'Embedding distances from last 50 epochs',
                               'ylabel': 'Embedding Distances',
                               'last_50': True}

    logs_path = 'Training Plots/Training...'

    losses_plot_callback = CallbackPlot(folder_path=logs_path,
                                        plots_settings=(loss_plot_settings, last_50_plot_settings),
                                        title='Losses', share_x=False)

    acc_and_distances_plot_callback = CallbackPlot(folder_path=logs_path,
                                                   plots_settings=(
                                                       accuracy_plot_settings, distances_plot_settings),
                                                   title='Accuracy and Embedding Distances', share_x=True)

    save_logs_callback = CallbackSaveLogs(folder_path=logs_path)

    model = make_inception_resnet_v2_model()

    create_folder(logs_path)

    history = model.fit(x=train_dataset, validation_data=test_dataset,
                        validation_steps=8, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, callbacks=[losses_plot_callback,
                                                  acc_and_distances_plot_callback,
                                                  save_logs_callback])

    # saves model
    if save_model:
        model.save(logs_path + '/trained_model.h5', save_format='h5')

    timestamp_end = datetime.now().strftime('%d-%b-%y -- %H:%M:%S')

    # renames the training folder with the end-of-training timestamp
    root, _ = os.path.split(logs_path)
    os.rename(logs_path, root + '/' + 'Training Session - ' + timestamp_end)

    return model, history
