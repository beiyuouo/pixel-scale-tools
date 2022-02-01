import os
import numpy as np


def img_pixel(img, srcX, srcY):
    return img[clamp(srcY, 0, img.shape[0] - 1), clamp(srcX, 0, img.shape[1] - 1)]


def luma(pixel):
    sum = 1 + pixel[0] + pixel[1] + pixel[2]
    # print(pixel)
    return sum * (256 - pixel[3])


def clamp(x, min_val, max_val):
    return max(min(x, max_val), min_val)


def all_eq(B, *args):
    args = np.array(args)
    return np.all(args == B)


def none_eq(B, *args):
    for arg in args:
        if np.equal(B, arg).all():
            return False
    return True


def any_eq(B, *args):
    for arg in args:
        if np.equal(B, arg).all():
            return True
    return False
