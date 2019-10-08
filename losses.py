#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:58:39 2019

@author: pedro
"""
import tensorflow as tf
import numpy as np


from config import units_embeddings
from config import threshold
from config import distance_margin


def deconcatenate_output_tensor(tensor):
    '''
    Separates the anchor, positive and negative images from the concatenated tensor

    tensor: tensor of shape (batch_size, 3 * units_last_embedding)
    anchor_encoding, positive_encoding, negative_encoding: tensors of shape (batch_size, units_last_embedding)
    '''

    anchor_encoding = tensor[:, 0:  units_embeddings[-1]]
    positive_encoding = tensor[:,
                               units_embeddings[-1]:  units_embeddings[-1]*2]
    negative_encoding = tensor[:,
                               units_embeddings[-1]*2:  units_embeddings[-1]*3]

    return anchor_encoding, positive_encoding, negative_encoding


def calculate_distances(anchor_encoding, positive_encoding, negative_encoding):
    '''
    Computes the distance between the anchor and positive, and anchor and negative

    anchor_encoding, positive_encoding, negative_encoding: tensors of shape (batch_size, units_last_embedding)
    positive_distance, negative_distance: tensors of shape (batch_size)
    '''
    positive_distances = tf.reduce_sum(
        tf.square(anchor_encoding - positive_encoding), axis=-1)
    negative_distances = tf.reduce_sum(
        tf.square(anchor_encoding - negative_encoding), axis=-1)

    return positive_distances, negative_distances


def triplet_loss(y_true, y_pred):
    '''
    Computes the custom triplet loss function.

    y_pred: output of model, tensor of shape (batch_size, 3 * units_last_embedding)
    y_true: a dummy parameter, that is needed for the keras API to recognize the function as a custom loss function
    loss: a tensor of shape 1, containing the loss of the current batch
    '''

    # anchor_encoding, positive_encoding, negative_encoding = deconcatenate_output_tensor(y_pred)
    anchor_encoding = y_pred[0]
    positive_encoding = y_pred[1]
    negative_encoding = y_pred[2]

    positive_distances, negative_distances = calculate_distances(
        anchor_encoding, positive_encoding, negative_encoding)

    # computes the median loss of the batch
    loss = tf.reduce_mean(tf.maximum(
        (positive_distances - negative_distances + distance_margin), 0))

    return loss


def probability_logistic_loss(y_true, y_pred):
    '''
    Computes the custom probability logistic loss function.

    y_pred: output of model, tensor of shape (2, batch_size)
    y_true: a dummy parameter, that is needed for the keras API to recognize the function as a custom loss function
    loss: a tensor of shape 1, containing the loss of the current batch
    '''

    # recovers positive and negative probabilities from concatenated tensor
    # prob_positive_pair, prob_negative_pair, _ = tf.unstack(y_pred)
    prob_positive_pairs = y_pred[0]
    prob_negative_pairs = y_pred[1]

    # small epsilon for numerical stability
    epsilon = 1e-07

    loss = - tf.reduce_mean((tf.math.log(prob_positive_pairs + epsilon)) +
                            tf.math.log(1 - prob_negative_pairs + epsilon))/2

    # loss = tf.reduce_mean(tf.math.log(1 - prob_negative_pair))

    return loss


def pos_dist(y_true, y_pred):
    '''
    Computes the mean positive distance (anchor-to-positive embedding distance)
    so the positive distance can be shown during training.
    '''

    positive_distances = y_pred[2]

    average_positive_distance = tf.reduce_mean(positive_distances)

    return average_positive_distance


def neg_dist(y_true, y_pred):
    '''
    Computes the mean negative distance (anchor-to-negative embedding distance)
    so the negative distance can be shown during training.
    '''

    negative_distances = y_pred[3]

    average_negative_distance = tf.reduce_mean(negative_distances)

    return average_negative_distance


def pos_prob(y_true, y_pred):
    '''
    Computes the average of probabilities from the positive pairs (ideally,
    should be close to 1)
    '''

    prob_positive_pair = y_pred[0]

    avg_prob_positive_pair = tf.reduce_mean(prob_positive_pair)

    return avg_prob_positive_pair


def neg_prob(y_true, y_pred):
    '''
    Computes the average of probabilities from the negative pairs (ideally,
    should be close to 0)
    '''

    prob_negative_pair = y_pred[1]

    avg_prob_negative_pair = tf.reduce_mean(prob_negative_pair)

    return avg_prob_negative_pair


def accuracy(y_true, y_pred):
    '''
    Computes the custom accuracy.

    y_pred: output of model, tensor of shape (batch_size, 3 * units_last_embedding)
    y_true: a dummy parameter, that is needed for the keras API to recognize the function as a custom loss function
    '''

    anchor_encoding, positive_encoding, negative_encoding = deconcatenate_output_tensor(
        y_pred)

    positive_distances, negative_distances = calculate_distances(
        anchor_encoding, positive_encoding, negative_encoding)

    # checks if distances between positive examples are less than the threshold, and if negative examples are greater
    positive_predictions = tf.math.less(positive_distances, threshold)
    negative_predictions = tf.math.greater_equal(negative_distances, threshold)

    final_prediction = tf.concat(
        [positive_predictions, negative_predictions], axis=0)

    accuracy = tf.reduce_mean(tf.cast(final_prediction, tf.float32))

    return accuracy


def TPR_FPR_values(probabilities, threshold_resolution=200):
    '''
    Computes the TPR (true positive rate) and
    FPR (false positive rate) values for different thresholds
    '''

    # probabilities from positive pairs
    prob_pos_pairs = probabilities[0]

    # probabilities from negative pairs
    prob_neg_pairs = probabilities[1]

    # true positive rate
    TPRs = []
    # false positive rate
    FPRs = []

    for i in range(threshold_resolution + 1):

        threshold = i/threshold_resolution
        # computes the TPR and FPR for different thresholds
        TPR = tf.reduce_mean(
            tf.cast(tf.greater(prob_pos_pairs, threshold), tf.float32))
        FPR = tf.reduce_mean(
            tf.cast(tf.greater(prob_neg_pairs, threshold), tf.float32))

        TPRs.append(TPR)
        FPRs.append(FPR)

    TPRs = tf.stack(TPRs)
    FPRs = tf.stack(FPRs)

    TPRs = tf.reverse(TPRs, axis=[-1])
    FPRs = tf.reverse(FPRs, axis=[-1])

    return TPRs, FPRs


def ROC_custom_metric(y_true, y_pred):
    '''
    Computes the area under curve of the AOC plot.
    '''

    # gets only the probabilities from output tensor
    predictions = y_pred[:2]

    # get list containing true positive rate and false negative rate
    # values for different thresholds
    TPRs, FPRs = TPR_FPR_values(predictions, threshold_resolution=200)

    # computes the total area under the ROC curve using the trapezoidal method
    bases_sum = TPRs[:-1] + TPRs[1:]

    dxs = FPRs[1:] - FPRs[:-1]

    trapezoidal_areas = tf.math.multiply(bases_sum, dxs)/2

    auc = tf.reduce_sum(trapezoidal_areas)

    return auc
