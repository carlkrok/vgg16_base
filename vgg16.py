import save_load_model

from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
import h5py


WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def vgg16():


    K.set_image_dim_ordering('tf')

    # Building VGG16 _base:

    img_input = Input(shape=(64, 64, 3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Dropout(0.5)(x)

    base_model = Model(img_input, x, name='vgg16')

    # load weights
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models', file_hash='6d6bbae143d832006294945121d1f1fc')
    base_model.load_weights(weights_path)

    #base_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    #base_model.summary()
   


    top_model = Flatten()(x)

    # Regression part
    fc1 = Dense(100, activation='tanh')(top_model)
    fc2 = Dense(50, activation='tanh')(fc1)
    fc3 = Dense(10, activation='tanh')(fc2)
    prediction = Dense(1, activation='tanh')(fc3)

    model = Model(inputs=img_input, outputs=prediction)


    for pretrained_layer in base_model.layers:
        if isinstance(pretrained_layer, Conv2D):
            for layer in model.layers:
                if pretrained_layer.name == layer.name:
                    layer.set_weights(pretrained_layer.get_weights())



    for this_layer in model.layers[:17]:
        this_layer.trainable = False

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()

    save_load_model.save_model(model, "vgg16_test")

    return 0;
