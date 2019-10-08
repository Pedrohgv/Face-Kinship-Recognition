

import tensorflow as tf

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

from losses import triplet_loss, probability_logistic_loss, pos_dist, neg_dist, pos_prob, neg_prob, ROC_custom_metric

# imports pre-trained Inception ResNet V2 model
from tensorflow.keras.applications import InceptionResNetV2

# imports from keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Lambda, Layer


def download_pre_trained_model():
    '''
    Downloads the pretrained model and saves it to disk
    '''

    # downloads the model, without the top layers
    pre_trained_model = InceptionResNetV2(
        include_top=False, weights='imagenet', input_shape=img_shape)

    # saves the model to disk
    pre_trained_model.save('Inception_ResNet_V2.h5', save_format='h5')

    return None


def create_fully_connected_layers(tensor, units_embeddings,
                                  use_dropout, activations,
                                  l2_regularization):
    '''
    Creates fully connected layers
    '''
    l2_regularizer = tf.keras.regularizers.l2(l=l2_regularization)

    x = tensor
    for units_embedding_layer, use_dropout_layer, l2_regularization_layer, activation in \
            zip(units_embeddings, use_dropout, l2_regularization, activations):

        l2_regularizer = tf.keras.regularizers.l2(l=l2_regularization_layer)
        x = Dense(units=units_embedding_layer, activation=activation,
                  kernel_regularizer=l2_regularizer)(x)

        if use_dropout_layer:
            x = Dropout(dropout_rate)(x)

    return x


def l2_norm(vectors, square_root=False):
    '''
    Computes the euclidean distance of 2 vectors
    '''
    norm = tf.reduce_sum(tf.square(vectors[0] - vectors[1]), axis=-1)

    if square_root:
        norm = tf.sqrt(norm)

    return norm


class L2Norm2Prob(Layer):
    '''
    Transforms a embedding distance between two tensors in a probability
    '''

    def __init__(self, **kwargs):

        super(L2Norm2Prob, self).__init__(**kwargs)

        # initializes the exponent and applies a constraint so it cannot be negative
        initializer = tf.keras.initializers.GlorotUniform()
        alpha_constraint = tf.keras.constraints.non_neg()
        regularizer = tf.keras.regularizers.l2(l=0.01)
        prob_constraint = tf.keras.constraints.MaxNorm(max_value=1)

        self.alpha = self.add_variable(
            name='alpha', initializer=initializer, constraint=alpha_constraint,
            regularizer=regularizer, trainable=True)
        self.beta = self.add_variable(
            name='beta', initializer='zeros',
            trainable=True)

    def call(self, x):

        probability = 1 - \
            tf.math.tanh(tf.math.add(
                tf.math.multiply(self.alpha, x), self.beta))

        # contraint to keep probabilities <= 1
        probability = tf.clip_by_value(
            probability, clip_value_min=0, clip_value_max=1)

        return probability

    def get_config(self):

        config = super(L2Norm2Prob, self).get_config()

        return config


def make_facenet_based_model():
    '''
    Builds a facenet based model
    '''

    print('Building model...')

    facenet = tf.keras.models.load_model('facenet_keras.h5')

    # triplet input
    input_anchor = Input(shape=img_shape, name='Anchor_input')
    input_positive = Input(shape=img_shape, name='Positive_input')
    input_negative = Input(shape=img_shape, name='Negative_input')

    # triplet output
    output_anchor = facenet(input_anchor)
    output_positive = facenet(input_positive)
    output_negative = facenet(input_negative)

    # lambda layer that computes the l2 norm between two embedding vectors
    l2_norm_layer = Lambda(l2_norm, arguments={
        'square_root': True}, name='L2_norm_layer')

    distance_positive_pair = l2_norm_layer([output_anchor, output_positive])
    distance_negative_pair = l2_norm_layer([output_anchor, output_negative])

    # layer that takes the computed distance between a pair of embeddings and outputs a probability of kinship
    prob_layer = L2Norm2Prob()

    prob_positive_pair = prob_layer(distance_positive_pair)
    prob_negative_pair = prob_layer(distance_negative_pair)

    # The output of the model (y_pred) must come in the form of a single tensor
    # (the probabilities and distances were stacked in a single tensor)
    # instead of a list because keras applies the loss function to all of the outputs
    # in a list separately. The positive and negative distances were added to the output
    # so they can be monitored using a custom metric
    stacked_output = Lambda(lambda tensors: tf.stack(tensors), name='Stack_layer')([prob_positive_pair, prob_negative_pair,
                                                                                    distance_positive_pair, distance_negative_pair])

    # builds the model
    model = Model(inputs=[input_anchor, input_positive, input_negative],
                  outputs=stacked_output)

    # set all layers to be not trainable before setting specific layers to be trainable
    for layer in facenet.layers:
        layer.trainable = False

    # only specific layers are to be trained
    for index in range(-1, -3, -1):
        facenet.layers[index].trainable = True

    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9)

    model.compile(optimizer=adam, loss=probability_logistic_loss,
                  metrics=[pos_prob, neg_prob, pos_dist, neg_dist, ROC_custom_metric])

    print('Model successfully built.')

    return model


def make_inception_resnet_v2_model():
    '''
    Builds the CNN model
    '''

    print('Building model...')

    # loads the inception v3 model, without the last classification layers
    inception_resnet_V2 = tf.keras.models.load_model('Inception_ResNet_V2.h5')

    # connects output of inception to new layers
    x = inception_resnet_V2.output
    # adds global average pooling for the last concatenated convolutional layers
    x = GlobalAveragePooling2D()(x)

    x = create_fully_connected_layers(x, units_embeddings,
                                      use_dropout, activations,
                                      l2_regularization)

    # base model
    base_model = Model(inputs=inception_resnet_V2.input, outputs=x)

    input_anchor = Input(shape=img_shape, name='input_anchor')
    input_positive = Input(shape=img_shape, name='input_positive')
    input_negative = Input(shape=img_shape, name='input_negative')

    output_anchor = base_model(input_anchor)
    output_positive = base_model(input_positive)
    output_negative = base_model(input_negative)

    # The output of the model (y_pred) must come in the form of a single tensor
    # (the anchor, positive and negative encodings were concatenated in a single tensor)
    # instead of a list because keras applies the loss function to all of the outputs
    # in a list separately.
    concatenated_output = Concatenate(
        axis=-1)([output_anchor, output_positive, output_negative])

    # builds the model
    model = Model(inputs=[input_anchor, input_positive, input_negative],
                  outputs=concatenated_output)

    # only the added top layers are to be trained
    for layer in inception_resnet_V2.layers:
        layer.trainable = False

    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9)

    model.compile(optimizer=adam, loss=triplet_loss,
                  metrics=[accuracy, pos_dist, neg_dist])

    print('Model successfully built.')

    return model
