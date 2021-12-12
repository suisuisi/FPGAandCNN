# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'


import pickle
import gzip
import cv2
import numpy as np


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def load_mnist_data(type='channel_last'):
    from keras.datasets import mnist
    from keras.utils import np_utils

    # input image dimensions
    nb_classes = 10
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if type == 'channel_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test


def save_history(history, path, columns=('loss', 'val_loss')):
    import matplotlib.pyplot as plt
    import pandas as pd
    s = pd.DataFrame(history.history)
    s.to_csv(path + '.csv')
    plt.plot(s[list(columns)])
    plt.savefig(path + '.png')
    plt.close()


def prepare_image_from_camera(im_path):
    img = cv2.imread(im_path)
    print('Read image: {} Shape: {}'.format(im_path, img.shape))

    # Take central part of image with size 224х224
    img = img[8:-8, 48:-48]
    print('Reduced shape: {}'.format(img.shape))

    # Convert to grayscale with human based formulae https://samarthbhargav.wordpress.com/2014/05/05/image-processing-with-python-rgb-to-grayscale-conversion/
    # Divider here is 16 for easier implementation of division in verilog for FPGA.
    # Colors are in BGR order
    gray = np.zeros(img.shape[:2], dtype=np.uint16)
    gray[...] = 3*img[:, :, 0].astype(np.uint16) + 8*img[:, :, 1].astype(np.uint16) + 5*img[:, :, 2].astype(np.uint16)
    gray //= 16

    # Invert color (don't need this)
    # gray = 255 - gray
    # show_image(gray.astype(np.uint8))

    # Rescale to 28x28 using mean pixel for each 8x8 block
    output_image = np.zeros((28, 28), dtype=np.uint8)
    for i in range(28):
        for j in range(28):
            output_image[i, j] = int(gray[i*8:(i+1)*8, j*8:(j+1)*8].mean())

    # Check dynamic range
    min_pixel = output_image.min()
    max_pixel = output_image.max()
    print('Min pixel: {}'.format(min_pixel))
    print('Max pixel: {}'.format(max_pixel))

    # Rescale dynamic range if needed (no Verilog implementation, so skip)
    if 0:
        if min_pixel != 0 or max_pixel != 255:
            if max_pixel == min_pixel:
                output_image[:, :] = 0
            else:
                output_image = 255 * (output_image.astype(np.float32) - min_pixel) / (max_pixel - min_pixel)
                output_image = output_image.astype(np.uint8)

    if 0:
        u = np.unique(output_image, return_counts=True)
        print(u)

    # Check image (rescaled x10 times)
    # show_resized_image(output_image, 280, 280)
    return output_image

