#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:58:39 2019

@author: pedro
"""
import tensorflow as tf


from config import units_embeddings

from config import threshold

from config import distance_margin

def deconcatenate_output_tensor(tensor):
    
# =============================================================================
#     Separates the anchor, positive and negative images from the concatenated tensor
#     
#     tensor: tensor of shape (batch_size, 3 * units_last_embedding)
#     anchor_encoding, positive_encoding, negative_encoding: tensors of shape (batch_size, units_last_embedding)
# =============================================================================
    
    anchor_encoding = tensor[:, 0 :  units_embeddings[-1]]
    positive_encoding = tensor[:,  units_embeddings[-1] :  units_embeddings[-1]*2]
    negative_encoding = tensor[:,  units_embeddings[-1]*2 :  units_embeddings[-1]*3]
    
    return anchor_encoding, positive_encoding, negative_encoding


def calculate_distances(anchor_encoding, positive_encoding, negative_encoding):
    
# =============================================================================
#     Computes the distance between the anchor and positive, and anchor and negative
#     
#     anchor_encoding, positive_encoding, negative_encoding: tensors of shape (batch_size, units_last_embedding)
#     positive_distance, negative_distance: tensors of shape (batch_size)
# =============================================================================
    positive_distances = tf.reduce_sum(tf.square(anchor_encoding - positive_encoding), axis = -1)
    negative_distances = tf.reduce_sum(tf.square(anchor_encoding - negative_encoding), axis = -1)
    
    return positive_distances, negative_distances


def triplet_loss(y_true, y_pred):
    
# =============================================================================
#     Computes the custom loss function.
#     
#     y_pred: output of model, tensor of shape (batch_size, 3 * units_last_embedding)
#     y_true: a dummy parameter, that is needed for the keras API to recognize the function as a custom loss function
#     loss: a tensor of shape 1, containing the loss of the current batch
# =============================================================================
    
    anchor_encoding, positive_encoding, negative_encoding = deconcatenate_output_tensor(y_pred)
    
    positive_distances, negative_distances = calculate_distances(anchor_encoding, positive_encoding, negative_encoding)
    
    # computes the median loss of the batch
    loss = tf.reduce_mean(tf.maximum((positive_distances - negative_distances + distance_margin), 0))
    
    return loss

def accuracy(y_true, y_pred):
    
# =============================================================================
#     Computes the custom accuracy.
#     
#     y_pred: output of model, tensor of shape (batch_size, 3 * units_last_embedding)
#     y_true: a dummy parameter, that is needed for the keras API to recognize the function as a custom loss function
# =============================================================================
    
    
    anchor_encoding, positive_encoding, negative_encoding = deconcatenate_output_tensor(y_pred)
    
    positive_distances, negative_distances = calculate_distances(anchor_encoding, positive_encoding, negative_encoding)
    
    # checks if distances between positive examples are less than the threshold, and if negative examples are greater
    positive_predictions = tf.math.less(positive_distances, threshold)
    negative_predictions = tf.math.greater_equal(negative_distances, threshold)
    
    final_prediction = tf.concat([positive_predictions, negative_predictions], axis = 0)
    
    accuracy = tf.reduce_mean(tf.cast(final_prediction, tf.float32))

    return accuracy




def pos_dist(y_true, y_pred):
    '''
    Computes the mean positive distance (anchor-to-positive embedding distance)
    so the positive distance can be shown during training, for the porpose of finding
    an optimal classifiation threshold
    '''
    
    anchor_encoding, positive_encoding, negative_encoding = deconcatenate_output_tensor(y_pred)
    
    positive_distances, _ = calculate_distances(anchor_encoding, positive_encoding, negative_encoding)
    
    average_positive_distance = tf.reduce_mean(positive_distances)
    
    return average_positive_distance

def neg_dist(y_true, y_pred):
    '''
    Computes the mean negative distance (anchor-to-negative embedding distance)
    so the negative distance can be shown during training, for the porpose of finding
    an optimal classifiation threshold
    '''
    
    anchor_encoding, positive_encoding, negative_encoding = deconcatenate_output_tensor(y_pred)
    
    _, negative_distances = calculate_distances(anchor_encoding, positive_encoding, negative_encoding)
    
    average_negative_distance = tf.reduce_mean(negative_distances)
    
    return average_negative_distance
    
    

    







