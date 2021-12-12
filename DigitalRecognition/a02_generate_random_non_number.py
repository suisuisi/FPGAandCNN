# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'


from a00_common_functions import *
from PIL import Image, ImageDraw
import numpy as np
import os
import cv2
import random
import math
import time


gradient_image = np.zeros((256, 256), dtype=np.uint8)
for i in range(256):
    gradient_image[i, :] = i


def rand_2coord(size, sp):
    if min(sp) < 5:
        if max(sp) > size - 5:
            x1 = random.randint(min(sp)+5,max(sp)-5)
        else:
            x1 = random.randint(max(sp) + 5, size)
    elif (max(sp) > size - 5):
        if ((random.randint(0,1)==0)&(min(sp)!=max(sp))&((max(sp)-5)>(min(sp)+5))):
            x1 = random.randint(min(sp)+5,max(sp)-5)
        else:
            x1 = random.randint(0, min(sp) - 5)
    elif ((max(sp)-5)<=(min(sp)+5)):
        if ((random.randint(0, 1) == 0) & (min(sp) != max(sp))):
            x1 = random.randint(0, min(sp) - 5)
        else:
            x1 = random.randint(max(sp) + 5, size)
    else:
        a=random.randint(0, len(sp))
        if (a==0):
            x1 = random.randint(0, min(sp) - 5)
        elif (a==1):
            x1 = random.randint(max(sp) + 5, size)
        else:
            x1 = random.randint(min(sp)+5,max(sp)-5)
    return (x1)


def get_random_mirror(image):
    # all possible mirroring and flips
    # (in total there are only 8 possible configurations)
    # image must be square for correct output
    mirror = random.randint(0, 1)
    if mirror == 1:
        # flipud
        image = image[::-1, :]
    angle = random.randint(0, 3)
    if angle != 0:
        image = np.rot90(image, k=angle)
    return image


def generate_random_non_number():

    def one_color(size):
       img = np.zeros(size, dtype=np.uint8)
       img[...] = random.randint(0, 255)
       return img

    def gradient(size):
        global gradient_image
        start_0 = random.randint(0, gradient_image.shape[0] - size[0])
        start_1 = random.randint(0, gradient_image.shape[1] - size[1])
        end_0 = random.randint(start_0 + size[0], gradient_image.shape[0])
        end_1 = random.randint(start_1 + size[1], gradient_image.shape[1])
        image = gradient_image[start_0:end_0, start_1:end_1].copy()
        image = cv2.resize(image, size, cv2.INTER_LANCZOS4)
        image = get_random_mirror(image)
        return image

    def ellipse(size):
        img = np.zeros(size, dtype=np.uint8)
        img[...] = random.randint(0, 100)
        color = random.randint(150, 255)
        center_0 = random.randint(0, size[0])
        center_1 = random.randint(0, size[1])
        r0 = random.randint(0, center_0)
        r1 = random.randint(0, center_1)
        cv2.ellipse(img, (center_0, center_1), (r0, r1), 0, 0, 360, color, -1)
        return img

    def rectangle(size):
        font = random.randint(0, 255)
        if (font <= 1):
            color = random.randint(font + 1, 255)
        elif (font >= 254):
            color = random.randint(0, font - 1)
        else:
            if (random.randint(0, 1) == 0):
                color = random.randint(0, font - 1)
            else:
                color = random.randint(font + 1, 255)
        image = Image.new("L", size, font)
        draw = ImageDraw.Draw(image)
        x0 = random.randint(0, size[0])
        y0 = random.randint(0, size[0])
        x1 = rand_2coord(size[0], [x0])
        y1 = rand_2coord(size[0], [y0])
        draw.rectangle((x0, y0, x1, y1), fill=color)
        del draw
        return (image)


    size = (28, 28)
    img_mult = {
        0: one_color(size),
        1: gradient(size),
    }
    img = img_mult[random.randint(0, 1)]
    return img


if __name__ == '__main__':
    while 1:
        img = generate_random_non_number()
        show_resized_image(img, 280, 280)