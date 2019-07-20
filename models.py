

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

from losses import triplet_loss, accuracy, pos_dist, neg_dist

# imports from keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate


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
    for units_embedding_layer, use_dropout_layer, l2_regularization_layer, activation in zip(units_embeddings, use_dropout, l2_regularization, activations):
        l2_regularizer = tf.keras.regularizers.l2(l=l2_regularization_layer)
        x = Dense(units=units_embedding_layer, activation=activation,
                  kernel_regularizer=l2_regularizer)(x)

        if use_dropout_layer:
            x = Dropout(dropout_rate)(x)

    return x


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
    concatenated_output = Concatenate(axis=-1)([output_anchor, output_positive, output_negative])

    # builds the model
    model = Model(inputs=[input_anchor, input_positive, input_negative],
                  outputs=concatenated_output)

    # only the added top layers are to be trained
    for layer in inception_resnet_V2.layers:
        layer.trainable = False

    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9)

    model.compile(optimizer=adam, loss=triplet_loss, metrics=[accuracy, pos_dist, neg_dist])

    print('Model successfully built.')

    return model
