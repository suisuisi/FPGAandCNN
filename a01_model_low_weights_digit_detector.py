# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'


def keras_model_low_weights_digit_detector():
    from keras.models import Model
    from keras.layers import Input, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
    from keras.layers import Conv2D, MaxPooling2D, Activation

    use_bias = False

    inputs1 = Input((28, 28, 1))
    x = Conv2D(4, (3, 3), activation=None, padding='same', name='conv1', use_bias=use_bias)(inputs1)
    x = Activation('relu')(x)
    x = Conv2D(4, (3, 3), activation=None, padding='same', name='conv2', use_bias=use_bias)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(8, (3, 3), activation=None, padding='same', name='conv3', use_bias=use_bias)(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), activation=None, padding='same', name='conv4', use_bias=use_bias)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    x = Conv2D(16, (3, 3), activation=None, padding='same', name='conv5', use_bias=use_bias)(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), activation=None, padding='same', name='conv6', use_bias=use_bias)(x)
    x = Activation('relu')(x)
    x = GlobalMaxPooling2D()(x)
    x = Dense(11, activation=None, use_bias=use_bias)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inputs1, outputs=x)
    print(model.summary())
    return model

