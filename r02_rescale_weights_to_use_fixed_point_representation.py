# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'


'''
This code try to find the minimum and maximum possible value on each layer and rescale weights to stay in [-1, 1] range.
'''


from a00_common_functions import *
from a01_model_low_weights_digit_detector import keras_model_low_weights_digit_detector
import glob
import os
import random

# Coefficient to make safe gap for found range to prevent overflow. Lower - less safe, higher - more rounding error.
GAP_COEFF = 1.1

gpu_use = 0
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


def bbox1(img):
    a = np.where(img < 10)
    try:
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    except:
        bbox = 0, img.shape[0]-1, 0, img.shape[1]-1
    return bbox


def random_rotate(image, max_angle):
    cols = image.shape[1]
    rows = image.shape[0]

    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return dst


def add_random_noize(image, prob):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if random.uniform(0, 1) < prob:
                image[i, j] = random.randint(0, 255)
    return image


def augment_single_image(img, class1):
    # Random rotate
    img_rotated = random_rotate(img.copy(), 10)

    # Random crop (only for non-background)
    if class1[10] != 1:
        img = np.zeros((32, 32))
        img[...] = 255
        img[2:-2, 2:-2] = img_rotated
        bb = bbox1(img)
        start_0 = random.randint(0, bb[0])
        end_0 = random.randint(bb[1] + 1, img.shape[0])
        start_1 = random.randint(0, bb[2])
        end_1 = random.randint(bb[3] + 1, img.shape[1])
        subimg = img[start_0:end_0, start_1:end_1].copy()
        interp_type = random.choice([cv2.INTER_LANCZOS4, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST])
        sub_enlarge = cv2.resize(subimg, (28, 28), interp_type)
    else:
        sub_enlarge = img_rotated

    # Random intensity change
    rand_intensity = random.randint(-80, 80)
    sub_enlarge = sub_enlarge.astype(np.int16) + rand_intensity
    sub_enlarge[sub_enlarge < 0] = 0
    sub_enlarge[sub_enlarge > 255] = 255
    sub_intencity_change = sub_enlarge.astype(np.uint8)

    # Random noize
    sub_noize = add_random_noize(sub_intencity_change, random.uniform(0, 0.1))

    img = sub_noize.copy()
    if 0:
        show_resized_image(img, 280, 280)
        show_resized_image(subimg, 280, 280)
        show_resized_image(sub_enlarge, 280, 280)
        show_resized_image(sub_intencity_change, 280, 280)
        show_resized_image(img, 280, 280)
        print(class1)

    return img


def prepare_imageset():
    # Part 1 (MNIST dataset inverted and augmented)
    X_train, Y_train, X_test, Y_test = load_mnist_data(type='channel_last')
    # Append class 10 for background
    Y_train = np.concatenate((Y_train, np.zeros((Y_train.shape[0], 1))), axis=1)
    Y_test = np.concatenate((Y_test, np.zeros((Y_test.shape[0], 1))), axis=1)
    X_data = np.concatenate((X_train, X_test), axis=0)
    Y_data = np.concatenate((Y_train, Y_test), axis=0)

    # Invert images
    X_data = 255. - X_data

    # Augment images
    for i in range(X_data.shape[0]):
        X_data[i, :, :, 0] = augment_single_image(X_data[i, :, :, 0], Y_data[i])

    # Part 2 (real images from camera)
    expected_answ = []
    files = glob.glob('./dataset/train/*/*.png')
    image_list = []
    for f in files:
        answ = int(os.path.basename(os.path.dirname(f)))
        expected_answ.append(answ)
        output_image = cv2.imread(f, 0)
        image_list.append(output_image)
    image_list = np.expand_dims(image_list, axis=3)

    X_data = np.concatenate((X_data, image_list), axis=0)
    X_data = np.array(X_data, dtype=np.float32) / 256.
    return X_data


def rescale_weights(model, layer_num, coeff):
    w = model.layers[layer_num].get_weights()
    model.layers[layer_num].set_weights(w / coeff)
    return model


# Current code only works if model has no bias in any layer!
def get_min_max_for_model(model):
    from keras.models import Model

    full_input = prepare_imageset()
    print('Input data to check: {}'.format(full_input.shape))

    reduction_koeffs = dict()
    for i in range(len(model.layers)):
        layer = model.layers[i]
        print(layer.name)
        w1 = layer.get_weights()
        if len(w1) > 0:
            submodel = Model(inputs=model.inputs, outputs=layer.output)
            print(submodel.summary())
            out = submodel.predict(full_input)
            # Ищем максимум среди выхода и весов. Веса не должны превышать 1.0 в том числе.
            red_coeff = GAP_COEFF*max(abs(out.min()), abs(out.max()), abs(w1[0].min()), abs(w1[0].max()))
            print('Shape for submodel: {} Min out value: {} Max out value: {}'.format(out.shape, out.min(), out.max()))
            print('Min weights value: {} Max weights value: {}'.format(w1[0].min(), w1[0].max()))
            print('Reduction koeff: {}'.format(red_coeff))
            model = rescale_weights(model, i, red_coeff)
            reduction_koeffs[i] = red_coeff
    print('Reduction koeffs: ', reduction_koeffs)
    return model, reduction_koeffs


if __name__ == '__main__':
    model = keras_model_low_weights_digit_detector()
    model.load_weights('weights/keras_model_low_weights_digit_detector.h5')
    model, reduction_koeffs = get_min_max_for_model(model)

    overall_reduction_rate = 1.0
    for i in sorted(reduction_koeffs.keys()):
        print('Layer {} reduction coeff: {}'.format(i, reduction_koeffs[i]))
        overall_reduction_rate *= reduction_koeffs[i]
    print('Overall reduction rate: {}'.format(overall_reduction_rate))

    output_model_file = 'weights/keras_model_low_weights_digit_detector_rescaled.h5'
    model.save(output_model_file)
    print('Fixed model weights saved in {} file'.format(output_model_file))
