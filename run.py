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
import cv2

# import some hyperparameters
from config import img_shape

from config import batch_size
from config import steps_per_epoch
from config import epochs
from config import learning_rate

from config import set_seed_datagen
from config import seed_datagen

from config import units_embeddings
from config import activations
from config import use_dropout
from config import dropout_rate
from config import l2_regularization

import losses
import models

# import some functions
from input_manager import process_data, make_triplet_dataset, pre_processing
from training_log_callbacks import CallbackPlot, CallbackSaveLogs, create_folder
from losses import triplet_loss, accuracy, pos_dist, neg_dist
from models import create_fully_connected_layers, make_inception_resnet_v2_model, make_facenet_based_model

# imports from keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate

# import plot_model function from keras
from tensorflow.keras.utils import plot_model

# import the Dataset API
from tensorflow.data import Dataset


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
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPU's have been initialized
            print(e)

    return None


def train(save_model=False):

    train_families, test_families, train_positive_relations, test_positive_relations = process_data(
        set_seed=set_seed_datagen, seed=seed_datagen)

    # constructs training data pipeline
    train_dataset = make_triplet_dataset(
        train_families, train_positive_relations)

    # constructs test data pipeline
    test_dataset = make_triplet_dataset(test_families, test_positive_relations)

    # dictionaries containing metadata for plotting during training
    loss_plot_settings = {'variables': {'loss': 'Training loss',
                                        'val_loss': 'Validation loss'},
                          'title': 'Losses',
                          'ylabel': 'Epoch Loss',
                          'last_50': False}

    roc_plot_settings = {'variables': {'ROC_custom_metric': 'Training AUC',
                                       'val_ROC_custom_metric': 'Validation AUC'},
                         'title': 'ROC - AUC',
                         'ylabel': 'AUC',
                         'last_50': False}

    probabilities_plot_settings = {'variables': {'pos_prob': 'Training positive probabilities',
                                                 'neg_prob': 'Training negative probabilities',
                                                 'val_pos_prob': 'Validation positive probabilities',
                                                 'val_neg_prob': 'Validation negative probabilities'},
                                   'title': 'Probabilities',
                                   'ylabel': 'Probabilities',
                                   'last_50': False}

    distances_plot_settings = {'variables': {'pos_dist': 'Training positive distances',
                                             'neg_dist': 'Training negative distances',
                                             'val_pos_dist': 'Validation positive distances',
                                             'val_neg_dist': 'Validation negative distances'},
                               'title': 'Embedding Distances',
                               'ylabel': 'Embedding distances',
                               'last_50': False}

    # folder path for creating the folder that will contain the training data
    logs_path = 'Training Plots/Training...'

    # creation of callback objects
    losses_and_roc_plot_callback = CallbackPlot(folder_path=logs_path,
                                                plots_settings=(
                                                    loss_plot_settings, roc_plot_settings),
                                                title='Losses and ROC', share_x=True)

    probs_and_dists_plot_callback = CallbackPlot(folder_path=logs_path,
                                                 plots_settings=(
                                                     probabilities_plot_settings, distances_plot_settings),
                                                 title='Probabilities and Embedding Distances', share_x=True)

    save_logs_callback = CallbackSaveLogs(folder_path=logs_path)

    # creation of model
    model = make_facenet_based_model()

    # creation of folder for saving the training data
    create_folder(logs_path)

    # train the model
    history = model.fit(x=train_dataset, validation_data=test_dataset,
                        validation_steps=8, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, callbacks=[losses_and_roc_plot_callback,
                                                  probs_and_dists_plot_callback,
                                                  save_logs_callback])

    # saves model
    if save_model:
        print('\nSaving model...')
        model.save(logs_path + '/trained_model.h5', save_format='h5')
        model.save_weights('{}/saved_weights.h5'.format(logs_path))
        print('\nModel successfully saved')

    # saves model layout
    print('\nSaving model layout')
    plot_model(model, to_file=logs_path + '/model.png',
               rankdir='LR', show_shapes=True)
    print('\nModel layout successfully saved')

    timestamp_end = datetime.now().strftime('%d-%b-%y -- %H:%M:%S')

    # renames the training folder with the end-of-training timestamp
    root, _ = os.path.split(logs_path)
    os.rename(logs_path, root + '/' + 'Training Session - ' + timestamp_end)

    return model, history


def make_submission_file(folder_name):
    '''
    Creates a submission file
    '''

    # creates a dataframe with images to test
    pairs = pd.read_csv('sample_submission.csv', usecols=[
        'img_pair'], squeeze=True)
    num_pairs = pairs.size
    images = pairs.str.split(pat='-', expand=True)

    # load model
    """ model = tf.keras.models.load_model('Training Plots/{}/trained_model.h5'.format(folder_name),
                                       custom_objects={'L2Norm2Prob': models.L2Norm2Prob, 'probability_logistic_loss': losses.probability_logistic_loss,
                                                       'pos_prob': losses.pos_prob, 'neg_prob': losses.neg_prob,
                                                       'pos_dist': losses.pos_dist, 'neg_dist': losses.neg_dist,
                                                       'ROC_custom_metric': losses.ROC_custom_metric}) """

    model = make_facenet_based_model()
    model.load_weights(
        'Training Plots/{}/saved_weights.h5'.format(folder_name))

    def prediction_input_generator(images):

        for i, img_pair in images.iterrows():
            img_1 = pre_processing(cv2.imread('test/' + img_pair.iloc[0]))
            img_2 = pre_processing(cv2.imread('test/' + img_pair.iloc[1]))

            # here, the 3th entry (Negative_input) is not necessary to compute the predictions, but it is needed because the model takes 3 inputs
            yield {'Anchor_input': img_1, 'Positive_input': img_2, 'Negative_input': img_1}

    dataset = Dataset.from_generator(lambda: prediction_input_generator(images),
                                     output_types=({'Anchor_input': tf.float32, 'Positive_input': tf.float32, 'Negative_input': tf.float32}))
    dataset = dataset.batch(1)

    predictions = model.predict(x=dataset, steps=num_pairs)
    predictions = predictions[0::4]

    dataframe = pd.read_csv('sample_submission.csv')
    dataframe.is_related = predictions

    dataframe.to_csv(
        'Training Plots/{}/submission.csv'.format(folder_name), index=False)

    # return every 4th element of output array, because model has 4 outputs (this was made this way because of training and training metrics)
    return predictions
