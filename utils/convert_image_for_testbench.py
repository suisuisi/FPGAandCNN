# coding: utf-8

from a00_common_functions import *
import os
import numpy as np
import glob
import math
import cv2


def convert_to_binary(float_val, precision, only_dot=0):
    up = math.floor(float_val)
    down = float_val - up
    up_binary_str = "{:b}".format(up)

    p = precision
    start = down
    down_binary_str = ''
    while(p > 0):
        if start * 2 >= 1:
            down_binary_str += '1'
            start *= 2
            start -= 1
        else:
            down_binary_str += '0'
            start *= 2
        p -= 1
    if only_dot == 0:
        return up_binary_str + '.' + down_binary_str
    return down_binary_str


def convert_to_normalized_form(value, reduction=10, precision=12):
    sign = 0
    ret = value
    if ret < 0:
        sign = 1
        ret = abs(ret)

    dec = ret / reduction
    bin = convert_to_binary(dec, precision, 1)
    return sign, dec, bin


def generate_test_image_memory_verilog(image, answer, precision, out_file, type):
    out = open(out_file, "w")
    out.write("// Precision: {}\n".format(precision))
    out.write("// Image size: {}x{}\n".format(image.shape[0], image.shape[1]))
    out.write("// Answer: {}\n\n".format(answer))

    out.write("initial\n")
    out.write("begin\n")
    total = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sign, dec, bin1 = convert_to_normalized_form(image[i][j].copy(), 1, precision - 1)
            sgn = ' '
            if sign == 1:
                sgn = '-'
            if type == 'hex':
                hx = hex(int(bin1, 2))[2:].upper()
                out.write("\tstorage[{}] = {}{}'h{}; // {}\n".format(total, sgn, precision-1, hx, image[i][j]))
            else:
                out.write("\tstorage[{}] = {}{}'b{}; // {}\n".format(total, sgn, precision-1, bin1, image[i][j]))
            total += 1
    out.write("end\n")
    out.close()


def prepare_image(im_path):
    img = cv2.imread(im_path)

    img = img[8:-8, 48:-48]
    # print('Reduced shape: {}'.format(img.shape))

    gray = np.zeros(img.shape[:2], dtype=np.uint16)
    gray[...] = 3*img[:, :, 0].astype(np.uint16) + 8*img[:, :, 1].astype(np.uint16) + 5*img[:, :, 2].astype(np.uint16)
    gray //= 16

    output_image = np.zeros((28, 28), dtype=np.uint8)
    for i in range(28):
        for j in range(28):
            output_image[i, j] = int(gray[i*8:(i+1)*8, j*8:(j+1)*8].mean())

    min_pixel = output_image.min()
    max_pixel = output_image.max()
    # print('Min pixel: {}'.format(min_pixel))
    # print('Max pixel: {}'.format(max_pixel))

    # Check image (enlarge 10 times)
    # show_resized_image(output_image, 280, 280)
    return output_image


def get_image_set(path):
    # Real images from camera
    expected_answ = []
    files = glob.glob(path + 'dataset/test/*/*.png')
    image_list = []
    for f in files:
        answ = int(os.path.basename(os.path.dirname(f)))
        expected_answ.append(answ)
        output_image = prepare_image(f)
        image_list.append(output_image)
    image_list = np.expand_dims(image_list, axis=3)
    image_list = np.array(image_list, dtype=np.float32) / 256.
    return image_list, expected_answ


if __name__ == '__main__':
    use_image = 20
    bp = 11
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
    images, answers = get_image_set(ROOT_PATH)
    print('Total images read: {}. Image number for testbench: {}'.format(len(images), use_image))
    print('Bit precision: {} (with sign: {})'.format(bp, bp+1))
    out_path = ROOT_PATH + "verilog/initial_image_bitdepth_{}_img_number_{}_answ_{}.v".format(bp+1, use_image, answers[use_image])
    generate_test_image_memory_verilog(images[use_image], answers[use_image], bp+1, out_path, 'bin')
    print('Verilog generated in {}'.format(out_path))