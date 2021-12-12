# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'


'''
This code finds out which bit size for weight lead to zero classification error on fixed point test data
comparing with floating point test data. Start search from 8 bits up to 32 bits.
For our LWDD net optimum is on 10 bits.
'''

import os

gpu_use = 0
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import glob
import numpy as np
from a00_common_functions import *
from a01_model_low_weights_digit_detector import keras_model_low_weights_digit_detector
import math


# Note: We suppose that every Conv2D layer has type "same"
# In Tensorflow weight matrices already transposed
def my_convolve(input, kernel):
    output = np.zeros((input.shape[0], input.shape[1]))
    zero_pad = np.zeros((input.shape[0] + 2, input.shape[1] + 2))
    zero_pad[1:-1, 1:-1] = input
    # kernel = np.flipud(kernel)
    # kernel = np.fliplr(kernel)
    for i in range(1, zero_pad.shape[0] - 1):
        for j in range(1, zero_pad.shape[1] - 1):
            sub = zero_pad[i-1:i+2, j-1:j+2]
            output[i-1, j-1] = np.sum(sub*kernel)
    return output


def my_convolve_fixed_point(input, kernel, bit):
    output = np.zeros((input.shape[0], input.shape[1]))
    zero_pad = np.zeros((input.shape[0] + 2, input.shape[1] + 2))
    zero_pad[1:-1, 1:-1] = input
    # kernel = np.flipud(kernel)
    # kernel = np.fliplr(kernel)
    for i in range(1, zero_pad.shape[0] - 1):
        for j in range(1, zero_pad.shape[1] - 1):
            sub = zero_pad[i-1:i+2, j-1:j+2]
            output[i-1, j-1] = np.sum((sub*kernel).astype(np.int64))
    return output


def preprocess_forward(arr, val):
    arr1 = arr.copy().astype(np.float32)
    arr1 /= val
    return arr1


def convert_to_fix_point(arr1, bit):
    arr2 = arr1.copy().astype(np.float32)
    arr2[arr2 < 0] = 0.0
    arr2 = np.round(np.abs(arr2) * (2 ** bit))
    arr3 = arr1.copy().astype(np.float32)
    arr3[arr3 > 0] = 0.0
    arr3 = -np.round(np.abs(-arr3) * (2 ** bit))
    arr4 = arr2 + arr3
    return arr4.astype(np.int64)


def from_fix_point_to_float(arr, bit):
    return arr / (2 ** bit)


def compare_outputs(s1, s2, debug_info=True):
    if s1.shape != s2.shape:
        print('Shape of arrays is different! {} != {}'.format(s1.shape, s2.shape))
    s = np.abs(s1 - s2)
    size = 1
    for dim in np.shape(s): size *= dim
    if debug_info:
        print('Max difference: {}'.format(s.max()))
        print('Avg difference: {}'.format(s.mean()/size))


def dump_memory_structure_conv(arr, out_file):
    print('Dump memory structure in file: {}'.format(out_file))
    out = open(out_file, "w")
    total = 0
    for a in range(arr.shape[2]):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                out.write(str(total) + " LVL: {} X: {} Y: {} ".format(a, i, j) + str(arr[i, j, a]) + '\n')
                total += 1

    out.close()


def dump_memory_structure_dense(arr, out_file):
    print('Dump memory structure for dense layer in file: {}'.format(out_file))
    out = open(out_file, "w")
    total = 0
    print('Shape:', arr.shape)
    for j in range(arr.shape[0]):
         out.write(str(total) + " POS: {} ".format(j) + str(arr[j]) + '\n')
         total += 1

    out.close()


def print_first_pixel_detailed_calculation_dense(previous_layer_output, wgt_bit, bit_precizion):
    i = 10
    conv_my = 0
    for j in range(0, previous_layer_output.shape[0]):
        print('Pixel {}: {}'.format(j, int(previous_layer_output[j])))
        print('Weight {}: {}'.format(j, wgt_bit[j][i]))
        conv_my += np.right_shift((previous_layer_output[j]*wgt_bit[j][i]).astype(np.int64), bit_precizion)
        if j > 0 and j % 9 == 8:
            print('Current conv_my: {}'.format(conv_my))
    print('Result first pixel: {}'.format(conv_my))
    exit()


def print_first_pixel_detailed_calculation(previous_layer_output, wgt_bit, bit_precizion):
    i = 0
    x = 0
    y = 0
    conv_my = 0
    print('Point: {} X: {} Y: {}'.format(i, x, y))
    print('Weights shape: {}'.format(wgt_bit.shape))
    for j in range(wgt_bit.shape[2]):
        full_image = previous_layer_output[:, :, j]
        zero_pad = np.zeros((full_image.shape[0] + 2, full_image.shape[1] + 2))
        zero_pad[1:-1, 1:-1] = full_image
        pics = zero_pad[x+1-1:x+1+2, y+1-1:y+1+2].astype(np.int64)
        print('Pixel area 3x3 for [{}, {}]:'.format(x, y), pics)
        kernel = wgt_bit[:, :, j, i].copy()
        # Не надо переворачивать для TensorFlow
        # kernel = np.flipud(kernel)
        # kernel = np.fliplr(kernel)
        print('Weights {}: {}'.format(j, kernel))
        res = np.sum(np.right_shift((pics*kernel).astype(np.int64), bit_precizion))
        print('Convolution result {}: {}'.format(j, res))
        conv_my += res

    print('Overall result: {}'.format(conv_my))
    if conv_my[conv_my > 2 ** bit_precizion].any() or conv_my[conv_my < - 2 ** bit_precizion].any():
        print('Overflow! {}'.format(conv_my[conv_my > 2 ** bit_precizion]))
        exit()
    if conv_my < 0:
        conv_my = 0
    exit()


# bit_precizion - Точность фиксированной точки в битах
def go_mat_model(model, image, reduction_koeffs, bit_precizion=18, debug_info=True):

    weights = dict()
    level_out = dict()
    level_out_reduced = dict()
    level_id = 0
    dump_memory_struct = False
    print_pixel_calc = False

    for i in range(len(model.layers)):
        layer = model.layers[i]
        if debug_info:
            print('Layer name:', layer.name)

        # Convolution layers
        if 'Conv2D' in str(type(layer)):
            if debug_info:
                print('Layer type: {}'.format('Conv2D'))
            if level_id == 0:
                # We have real image at first layer
                previous_layer_output = image.copy()
            else:
                previous_layer_output = level_out[level_id - 1].copy()
            if debug_info:
                print('Input shape: {}'.format(previous_layer_output.shape))

            # Standard float convolution
            weights[level_id] = layer.get_weights()[0]
            if debug_info:
                print('Weights shape: {}'.format(weights[level_id].shape))
            level_out[level_id] = np.zeros((previous_layer_output.shape[0], previous_layer_output.shape[1], weights[level_id].shape[-1]))
            for i in range(weights[level_id].shape[-1]):
                conv_my = 0
                for j in range(weights[level_id].shape[-2]):
                    conv_my += my_convolve(previous_layer_output[:, :, j], weights[level_id][:, :, j, i])
                conv_my[conv_my < 0] = 0
                level_out[level_id][:, :, i] = conv_my

            # Convolution with fixed point
            if level_id == 0:
                koeff_weights = 1.0
                image_converted = preprocess_forward(image.copy(), koeff_weights)
                img_bit = convert_to_fix_point(image_converted.copy(), bit_precizion)
                previous_layer_output = img_bit.copy()
            else:
                previous_layer_output = level_out_reduced[level_id - 1].copy()

            # Weights reduction coeff - can be removed
            koeff_weights = 1.0
            weights_converted = preprocess_forward(weights[level_id].copy(), koeff_weights)
            wgt_bit = convert_to_fix_point(weights_converted.copy(), bit_precizion)

            level_out_reduced[level_id] = np.zeros_like(level_out[level_id])
            for i in range(wgt_bit.shape[-1]):
                conv_my = 0
                for j in range(wgt_bit.shape[-2]):
                    conv_my += my_convolve_fixed_point(previous_layer_output[:, :, j], wgt_bit[:, :, j, i], bit_precizion)
                conv_my[conv_my < 0] = 0
                conv_my = np.right_shift(conv_my.astype(np.int64), bit_precizion)
                if conv_my[conv_my > 2 ** bit_precizion].any() or conv_my[conv_my < - 2 ** bit_precizion].any():
                    print('Overflow level 1! {}'.format(conv_my[conv_my > 2 ** bit_precizion]))
                    print('Max is {}'.format(2 ** bit_precizion))
                    exit()
                level_out_reduced[level_id][:, :, i] = conv_my

            # Convert back to float for comparison
            checker_tmp = from_fix_point_to_float(level_out_reduced[level_id], bit_precizion)
            compare_outputs(level_out[level_id], checker_tmp, debug_info)
            if debug_info:
                print('Output shape: {}'.format(level_out[level_id].shape))

            if dump_memory_struct:
                dump_memory_structure_conv(level_out_reduced[level_id].astype(np.int64),
                                        "verilog/memory_structure_level_{}_conv.txt".format(level_id))

            # if layer.name == 'conv1':
            #    print_first_pixel_detailed_calculation(previous_layer_output, wgt_bit, bit_precizion)

            level_id += 1

        # Global max pooling
        if 'GlobalMaxPooling2D' in str(type(layer)):
            if debug_info:
                print('Layer type: {}'.format('GlobalMaxPooling2D'))

            # Standard glob pool
            previous_layer_output = level_out[level_id - 1].copy()
            if debug_info:
                print('Input shape: {}'.format(previous_layer_output.shape))
            level_out[level_id] = []
            for i in range(previous_layer_output.shape[-1]):
                level_out[level_id].append(previous_layer_output[:, :, i].max())
            level_out[level_id] = np.array(level_out[level_id])

            # Glob pool fixed point
            previous_layer_output = level_out_reduced[level_id - 1].copy()
            level_out_reduced[level_id] = []
            for i in range(previous_layer_output.shape[-1]):
                level_out_reduced[level_id].append(previous_layer_output[:, :, i].max())
            level_out_reduced[level_id] = np.array(level_out_reduced[level_id])
            if debug_info:
                print('Output shape: {}'.format(level_out[level_id].shape))
            level_id += 1

        # MaxPooling2D layers
        elif 'MaxPooling2D' in str(type(layer)):
            if debug_info:
                print('Layer type: {}'.format('MaxPooling2D'))
            previous_layer_output = level_out[level_id - 1].copy()
            if debug_info:
                print('Input shape: {}'.format(previous_layer_output.shape))

            # Level 3 (Pooling)
            level_out[level_id] = np.zeros((previous_layer_output.shape[0] // 2, previous_layer_output.shape[1] // 2, previous_layer_output.shape[2]))
            for i in range(previous_layer_output.shape[-1]):
                conv_my = np.zeros((previous_layer_output.shape[0] // 2, previous_layer_output.shape[1] // 2))
                for j in range(0, previous_layer_output.shape[0], 2):
                    for k in range(0, previous_layer_output.shape[1], 2):
                        conv_my[j // 2, k // 2] = max(previous_layer_output[j, k, i],
                                                      previous_layer_output[j + 1, k, i],
                                                      previous_layer_output[j, k + 1, i],
                                                      previous_layer_output[j + 1, k + 1, i])
                level_out[level_id][:, :, i] = conv_my

            previous_layer_output = level_out_reduced[level_id - 1]
            # Level 3 (Pooling reduced)
            level_out_reduced[level_id] = np.zeros_like(level_out[level_id])
            for i in range(previous_layer_output.shape[-1]):
                conv_my = np.zeros((previous_layer_output.shape[0] // 2, previous_layer_output.shape[1] // 2))
                for j in range(0, previous_layer_output.shape[0], 2):
                    for k in range(0, previous_layer_output.shape[1], 2):
                        conv_my[j // 2, k // 2] = max(previous_layer_output[j, k, i],
                                                      previous_layer_output[j + 1, k, i],
                                                      previous_layer_output[j, k + 1, i],
                                                      previous_layer_output[j + 1, k + 1, i])
                level_out_reduced[level_id][:, :, i] = conv_my
            if debug_info:
                print('Output shape: {}'.format(level_out[level_id].shape))
            level_id += 1

        # Dense layer (Softmax activation)
        elif 'Dense' in str(type(layer)):
            if debug_info:
                print('Layer type: {}'.format('Dense'))
            previous_layer_output = level_out[level_id - 1]
            if debug_info:
                print('Input shape: {}'.format(previous_layer_output.shape))

            weights[level_id] = layer.get_weights()[0]
            if debug_info:
                print('Weights shape: {}'.format(weights[level_id].shape))

            level_out[level_id] = []
            for i in range(weights[level_id].shape[1]):
                conv_my = 0
                for j in range(weights[level_id].shape[0]):
                    conv_my += previous_layer_output[j] * weights[level_id][j][i]
                level_out[level_id].append(conv_my)
            level_out[level_id] = np.array(level_out[level_id])

            # Softmax part
            out = level_out[level_id].copy()
            maxy = out.max()
            for i in range(out.shape[0]):
                out[i] = math.exp(out[i] - maxy)
            sum = out.sum()
            for i in range(out.shape[0]):
                out[i] /= sum

            # Dense reduced, max instead of softmax
            previous_layer_output = level_out_reduced[level_id - 1]
            level_out_reduced[level_id] = []

            weights_converted = weights[level_id].copy()
            wgt_bit = convert_to_fix_point(weights_converted.copy(), bit_precizion)

            for i in range(weights[level_id].shape[1]):
                conv_my = 0
                for j in range(weights[level_id].shape[0]):
                    conv_my += (previous_layer_output[j] * wgt_bit[j][i]).astype(np.int64)
                conv_my = np.right_shift(conv_my.astype(np.int64), bit_precizion)
                if conv_my[conv_my > 2 ** bit_precizion].any() or conv_my[conv_my < - 2 ** bit_precizion].any():
                    print('Overflow! {}'.format(conv_my[conv_my > 2 ** bit_precizion]))
                    exit()
                level_out_reduced[level_id].append(conv_my)
            level_out_reduced[level_id] = np.array(level_out_reduced[level_id])
            checker_tmp = from_fix_point_to_float(level_out_reduced[level_id], bit_precizion)
            compare_outputs(level_out[level_id], checker_tmp, debug_info)

            level_out[level_id] = out
            if debug_info:
                print('Output shape: {}'.format(level_out[level_id].shape))

            if dump_memory_struct:
                dump_memory_structure_dense(level_out_reduced[level_id].astype(np.int64),
                                        "verilog/memory_structure_level_{}_dense.txt".format(level_id))

            if layer.name == 'dense_1' and print_pixel_calc:
                print(wgt_bit.max())
                print(wgt_bit.min())
                print(weights[level_id].min())
                print(weights[level_id].max())
                print_first_pixel_detailed_calculation_dense(previous_layer_output, wgt_bit, bit_precizion)
            level_id += 1

        if debug_info:
            print('')

    level_id -= 1
    raw2 = from_fix_point_to_float(level_out_reduced[level_id], bit_precizion)
    amax2 = np.argmax(level_out_reduced[level_id])

    if dump_memory_struct:
        print('Exit because of dump memory. Set it to false to go next!')
        exit()
    return amax2, level_out[level_id], level_out_reduced[level_id], raw2


def get_error_rate(a1, a2):
    miss = 0
    for i in range(len(a1)):
        if a1[i] != a2[i]:
            miss += 1
    print('Error rate: {}%'.format(round(100*miss/len(a1), 2)))
    return miss


def get_image_set():
    # Part 2 (real images from camera)
    expected_answ = []
    files = glob.glob('./dataset/test/*/*.png')
    image_list = []
    for f in files:
        answ = int(os.path.basename(os.path.dirname(f)))
        expected_answ.append(answ)
        output_image = prepare_image_from_camera(f)
        image_list.append(output_image)
    image_list = np.expand_dims(image_list, axis=3)
    image_list = np.array(image_list, dtype=np.float32) / 256.
    return image_list, expected_answ


# This function works slow, so it should be run once to find optimal bit
def get_optimal_bit_for_weights():
    print('Read model...')
    use_cache = 1
    cache_path = 'weights/optimal_bit.pklz'
    if not os.path.isfile(cache_path) or use_cache == 0:
        model = keras_model_low_weights_digit_detector()

        # We read already reduced weights. We don't need to fix them any way
        model.load_weights('weights/keras_model_low_weights_digit_detector_rescaled.h5')
        images, answers = get_image_set()

        print('Classify images...')
        keras_out = model.predict(images)
        res_keras_array = []
        for i in range(keras_out.shape[0]):
            res_keras_array.append(np.argmax(keras_out[i]))
        print('Keras result: ', res_keras_array)

        for bp in range(8, 32):
            print('Start error precision: {}'.format(bp))
            res_model_array = []
            for i in range(len(images)):
                model_out, raw_output_v1, raw_output, raw_output_2 = go_mat_model(model, images[i], 0, bp, debug_info=False)
                res_model_array.append(model_out)
                print("Image number: {} Result: {} vs {}".format(i, res_keras_array[i], res_model_array[i]))
                if res_keras_array[i] != res_model_array[i]:
                    print('Keras[{}]: {}'.format(keras_out[i].shape, keras_out[i]))
                    print('Model (must be exact as Keras): {}'.format(raw_output_v1))
                    print('Model: {}'.format(raw_output))
                    print('Model: {}'.format(raw_output_2))
            miss = get_error_rate(res_keras_array, res_model_array)
            if miss == 0:
                save_in_file(bp, cache_path)
                return bp
        return -1
    else:
        return load_from_file(cache_path)


if __name__ == '__main__':
    bp = get_optimal_bit_for_weights()
    if bp > 0:
        print('Optimal bit size for weights (sign bit is not included) is: {}'.format(bp))
    else:
        print('Impossible to find optimal bit!')
