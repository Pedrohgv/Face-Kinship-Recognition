# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:58:33 2019

@author: btq9
"""

import os
import pandas as pd
import cv2
import random
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input 

# training set size
number_train_families = 377

def process_data(set_seed = False, seed = 1):

# =============================================================================
#     Process the input data.
#     Returns:
#         train_families: series containing all families to be used in training
#         test_families: series containing all families to be used in testing
#         train_positive_relations: series containing all pairs of kinship relation to be used in training
#         test_positive_relations: series containing all pairs of kinship relation to be used in testing
# =============================================================================
    
    # loads the positive relations of kinship
    positive_relations = pd.read_csv('train_relationships.csv')
    
    # remove relations that can't be found on the images folder
    positive_relations = check_families(positive_relations)
    
    # creates a series that contains the families of all relations
    relations_families = positive_relations.p1.str.split('/').str.get(0)
    
    # gets list of families from folder names
    families = os.listdir('train')
    families = pd.Series(families)
    
    # randomly shuffle 'families' and separates train and test sets
    if set_seed:
        families = families.sample(frac = 1, random_state = seed)
    else:
        families = families.sample(frac = 1)
    train_families = families.iloc[0:number_train_families]
    test_families = families.iloc[number_train_families:]
    
    # separates the positive relations in train and test dataframes
    train_positive_relations = positive_relations[relations_families.isin(train_families.values)]
    test_positive_relations = positive_relations[relations_families.isin(test_families.values)]
    
    return train_families, test_families, train_positive_relations, test_positive_relations

def check_families(positive_relations):
    
# =============================================================================
#     Removes kinship relations that are on the positive relations file and contains individuals that 
#     can't be found in the images folder.  
# =============================================================================
    
    indexes_to_remove = []
    # checks all the relations of the positive relations dataframe for individuals that can't be found
    for relation in positive_relations.index:
        if (not(os.path.exists('train/' + positive_relations.loc[relation,'p1'])) or \
            not(os.path.exists('train/' + positive_relations.loc[relation,'p2']))):
            indexes_to_remove.append(relation)
    
    # remove relations that couldn't be found        
    positive_relations = positive_relations.drop(labels = indexes_to_remove, axis = 0)
    
    return positive_relations

def get_triplet_img_folders(families, positive_relations, set_seed = False, seed = 1):
    
# =============================================================================
#     Get the paths of folders containing images from 3 different persons (anchor, positive and negative)
# =============================================================================
    
    if set_seed == True:
        positive_pair = positive_relations.sample(n = 1).sample(n = 2, axis = 1, random_state = seed)
    else: 
        positive_pair = positive_relations.sample(n = 1).sample(n = 2, axis = 1)
    
    # gets anchor and positive from positive_pair
    anchor = positive_pair.iloc[:,0].values[0]
    positive = positive_pair.iloc[:,1].values[0]
    
    # gets a negative candidate
    random_family = random.choice(families.values)
    random_person = random_family + '/' + random.choice(os.listdir('train/' + random_family))
    
    # checks if negative candidate really doesn't have a kinship relation with the anchor
    while(check_kinship(positive_relations, anchor, random_person) == True):
        random_family = random.choice(families.values)
        random_person = random_family + '/' + random.choice(os.listdir('train/' + random_family))
        
    # found a negative example
    negative = random_person
    
    # updates the paths with the root folder 'train'
    anchor = 'train/' + anchor
    positive = 'train/' + positive
    negative = 'train/' + negative
    
    return anchor, positive, negative

def get_triplet_img_paths(families, positive_relations):
 
# =============================================================================
#     Contruct a triplet of paths to anchor, positive and negative images
# =============================================================================
    
    # selects 3 folders containing images of 3 persons, to be the anchor, positive and negative examples
    anchor, positive, negative = get_triplet_img_folders(families, positive_relations, set_seed = True)
    
    # creates a list of images contained in each of the anchor, positive and negative folders
    list_of_anchor_images = os.listdir(anchor)
    list_of_positive_images = os.listdir(positive)
    list_of_negative_images = os.listdir(negative)
    
    # checks if any of the folders do not contain any images
    while([] in (list_of_anchor_images, list_of_positive_images, list_of_negative_images)):
        anchor, positive, negative = get_triplet_img_folders(families, positive_relations, set_seed = True)
        
        # creates a list of images contained in each of the anchor, positive and negative folders
        list_of_anchor_images = os.listdir(anchor)
        list_of_positive_images = os.listdir(positive)
        list_of_negative_images = os.listdir(negative)
        
    
    # builds the path for the randomly chosen images
    path_anchor_img = anchor + '/' + random.choice(list_of_anchor_images)
    path_positive_img = positive + '/' + random.choice(list_of_positive_images)
    path_negative_img = negative + '/' + random.choice(list_of_negative_images)
        
    return path_anchor_img, path_positive_img, path_negative_img


def make_triplet_dataset(families, positive_relations):
    
# =============================================================================
#     Dataset Generator that returns a random anchor, positive and negative images each time it is called
# =============================================================================
    while True:
        
        # generates random triplet
        path_anchor_img, path_positive_img, path_negative_img = get_triplet_img_paths(families, positive_relations)
        
        # loads and preprocess the images to be used in the in the algorithm 
        anchor_img = preprocess_input(cv2.imread(path_anchor_img)) # preprocessing does a (img/127.5) - 1 operation
        positive_img = preprocess_input(cv2.imread(path_positive_img))
        negative_img = preprocess_input(cv2.imread(path_negative_img))
        
        # 0 is just a dummy 'y_true' value, since true labels aren't required in triplet loss
        yield ({'input_anchor': anchor_img, 'input_positive': positive_img, 'input_negative': negative_img}, 0)

def check_kinship(positive_relations, anchor, negative_candidate):
    
# =============================================================================
#     Checks if anchor and negative candidate share a kinship relation
#     Returns True if there is a kin relation, and False otherwise 
# =============================================================================
    answer = ((positive_relations == [negative_candidate, anchor]).all(axis = 1).any() | \
              (positive_relations == [anchor, negative_candidate]).all(axis = 1).any())
    
    return answer








    
