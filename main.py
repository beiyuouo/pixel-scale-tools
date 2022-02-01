#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@File    :   main.py 
@Time    :   2022-02-01 10:55:43 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""
import os
import cv2
from core import *


def main():
    img_name = 'misery.png'
    img = cv2.imread(os.path.join('examples', img_name), cv2.IMREAD_UNCHANGED)
    cv2.imshow('img', img)
    img_scaled2x = mmpx_scale(img, 2)
    cv2.imwrite(
        os.path.join('examples',
                     img_name.split('.')[0] + '_scaled2x.' + img_name.split('.')[1]),
        img_scaled2x)
    cv2.imshow('img_scaled2x', img_scaled2x)

    img_scaled4x = mmpx_scale(img_scaled2x, 2)
    cv2.imwrite(
        os.path.join('examples',
                     img_name.split('.')[0] + '_scaled4x.' + img_name.split('.')[1]),
        img_scaled4x)
    cv2.imshow('img_scaled4x', img_scaled4x)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.waitKey(0)


if __name__ == '__main__':
    main()