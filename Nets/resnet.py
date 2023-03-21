from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, GlobalMaxPooling2D
from keras.layers.merge import add
from keras.activations import relu, softmax, sigmoid
from keras.models import Model
from keras import regularizers
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

def get_resnet(patch_size):

    # create the base pre-trained model
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(patch_size, patch_size, 3))
    # we add a global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add also a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # we have 2 classes only, so:
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # we freeze all the convolutional layers of ResNet50
    for layer in base_model.layers:
        layer.trainable = False

    # finally we compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model