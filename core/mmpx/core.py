#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@File    :   core\mmpx\core.py 
@Time    :   2022-02-01 14:22:19 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import os
import cv2
import numpy as np

from .utils import *


def scale(img, scale_factor=2):
    img = np.array(img)
    h, w, c = img.shape
    img_scaled = np.zeros((h * 2, w * 2, c), dtype=img.dtype)

    print(img.shape, img_scaled.shape)

    # for i in range(h):
    #     for j in range(w):
    #         img_scaled[i * scale_factor:(i + 1) * scale_factor,
    #                    j * scale_factor:(j + 1) * scale_factor] = img[i, j]

    for i in range(h):
        srcY, srcX = i, 0

        A, B, C = img_pixel(img, srcX - 1,
                            srcY - 1), img_pixel(img, srcX,
                                                 srcY - 1), img_pixel(img, srcX + 1, srcY - 1)
        D, E, F = img_pixel(img, srcX - 1,
                            srcY), img_pixel(img, srcX, srcY), img_pixel(img, srcX + 1, srcY)
        G, H, I = img_pixel(img, srcX - 1,
                            srcY + 1), img_pixel(img, srcX,
                                                 srcY + 1), img_pixel(img, srcX + 1, srcY + 1)
        Q, R = img_pixel(img, srcX - 2, srcY), img_pixel(img, srcX + 2, srcY)

        for j in range(w):
            srcX = j
            J, K, L, M = E.copy(), E.copy(), E.copy(), E.copy()

            # if none_eq(A, B, C, D, E, F, G, H):
            P = img_pixel(img, srcX, srcY - 2)
            S = img_pixel(img, srcX, srcY + 2)
            Bl, Dl, El, Fl, Hl, = luma(B), luma(D), luma(E), luma(F), luma(H)
            # 1:1 slope
            # if (D == B and D != H and D != F) and (El >= Dl or E == A) and any_eq(
            #         E, A, C, G) and (El < Dl or A != D or E != P or E != Q):
            #     J = D

            if (np.equal(D, B).all() and np.not_equal(D, H).any() and
                    np.not_equal(D, F).any()) and (El >= Dl or np.equal(E, A).all()) and any_eq(
                        E, A, C, G) and (El < Dl or np.not_equal(A, D).any() or np.not_equal(
                            E, P).any() or np.not_equal(E, Q).any()):
                J = D.copy()

            # if (B == F and B != D and B != H) and (El >= Bl or E == C) and any_eq(
            #         E, A, C, I) and (El < Bl or C != B or E != P or E != R):
            #     K = B

            if (np.equal(B, F).all() and np.not_equal(B, D).any() and
                    np.not_equal(B, H).any()) and (El >= Bl or np.equal(E, C).all()) and any_eq(
                        E, A, C, I) and (El < Bl or np.not_equal(C, B).any() or np.not_equal(
                            E, P).any() or np.not_equal(E, R).any()):
                K = B.copy()

            # if (H == D and H != B and H != F) and (El >= Hl or E == G) and any_eq(
            #         E, A, G, I) and (El < Hl or G != H or E != S or E != Q):
            #     L = H

            if (np.equal(H, D).all() and np.not_equal(H, B).any() and
                    np.not_equal(H, F).any()) and (El >= Hl or np.equal(E, G).all()) and any_eq(
                        E, A, G, I) and (El < Hl or np.not_equal(G, H).any() or np.not_equal(
                            E, S).any() or np.not_equal(E, Q).any()):
                L = H.copy()

            # if (F == H and F != B and F != D) and (El >= Fl or E == I) and any_eq(
            #         E, C, G, I) and (El < Fl or I != H or E != R or E != S):
            #     M = F

            if (np.equal(F, H).all() and np.not_equal(F, B).any() and
                    np.not_equal(F, D).any()) and (El >= Fl or np.equal(E, I).all()) and any_eq(
                        E, C, G, I) and (El < Fl or np.not_equal(I, H).any() or np.not_equal(
                            E, R).any() or np.not_equal(E, S).any()):
                M = F.copy()

            # intersection rules
            if np.not_equal(E, F).any() and all_eq(E, C, I, D, Q) and all_eq(
                    F, B, H) and np.not_equal(F, img_pixel(img, srcX + 3, srcY)).any():
                K, M = F.copy(), F.copy()
            if np.not_equal(E, D).any() and all_eq(E, A, G, F, R) and all_eq(
                    D, B, H) and np.not_equal(D, img_pixel(img, srcX - 3, srcY)).any():
                J, L = D.copy(), D.copy()
            if np.not_equal(E, H).any() and all_eq(E, G, I, B, P) and all_eq(
                    H, D, F) and np.not_equal(H, img_pixel(img, srcX, srcY + 3)).any():
                L, M = H.copy(), H.copy()
            if np.not_equal(E, B).any() and all_eq(E, A, C, H, S) and all_eq(
                    B, D, F) and np.not_equal(B, img_pixel(img, srcX, srcY - 3)).any():
                J, K = B.copy(), B.copy()
            if Bl < El and all_eq(E, G, H, I, S) and none_eq(E, A, D, C, F):
                J, K = B.copy(), B.copy()
            if Hl < El and all_eq(E, A, B, C, P) and none_eq(E, D, G, I, F):
                L, M = H.copy(), H.copy()
            if Fl < El and all_eq(E, A, D, G, Q) and none_eq(E, B, C, I, H):
                K, M = F.copy(), F.copy()
            if Dl < El and all_eq(E, C, F, I, R) and none_eq(E, B, A, G, H):
                J, L = D.copy(), D.copy()

            # 2:1 slope
            if np.not_equal(H, B).any():
                # if (H != A && H != E && H != C)
                if np.not_equal(H, A).any() and np.not_equal(H, E).any() and np.not_equal(
                        H, C).any():
                    # if (all_eq3(H, G, F, R) && none_eq2(H, D, src(&meta, srcX + 2, srcY - 1))) L = M;
                    if all_eq(H, G, F, R) and none_eq(H, D, img_pixel(img, srcX + 2, srcY - 1)):
                        L = M.copy()

                    # if (all_eq3(H, I, D, Q) && none_eq2(H, F, src(&meta, srcX - 2, srcY - 1))) M = L;
                    if all_eq(H, I, D, Q) and none_eq(H, F, img_pixel(img, srcX - 2, srcY - 1)):
                        M = L.copy()
                # if (B != I && B != G && B != E)
                if np.not_equal(B, I).any() and np.not_equal(B, G).any() and np.not_equal(
                        B, E).any():
                    # if (all_eq3(B, A, F, R) && none_eq2(B, D, src(&meta, srcX + 2, srcY + 1))) J = K;
                    if all_eq(B, A, F, R) and none_eq(B, D, img_pixel(img, srcX + 2, srcY + 1)):
                        J = K.copy()
                    # if (all_eq3(B, C, D, Q) && none_eq2(B, F, src(&meta, srcX - 2, srcY + 1))) K = J;
                    if all_eq(B, C, D, Q) and none_eq(B, F, img_pixel(img, srcX - 2, srcY + 1)):
                        K = J.copy()

            if np.not_equal(F, D).any():
                # if (D != I && D != E && D != C)
                if np.not_equal(D, I).any() and np.not_equal(D, E).any() and np.not_equal(
                        D, C).any():
                    # if (all_eq3(D, A, H, S) && none_eq2(D, B, src(&meta, srcX + 1, srcY + 2))) J = L;
                    if all_eq(D, A, H, S) and none_eq(D, B, img_pixel(img, srcX + 1, srcY + 2)):
                        J = L.copy()
                    # if (all_eq3(D, G, B, P) && none_eq2(D, H, src(&meta, srcX + 1, srcY - 2))) L = J;
                    if all_eq(D, G, B, P) and none_eq(D, H, img_pixel(img, srcX + 1, srcY - 2)):
                        L = J.copy()
                # if (F != E && F != A && F != G)
                if np.not_equal(F, E).any() and np.not_equal(F, A).any() and np.not_equal(
                        F, G).any():
                    # if (all_eq3(F, C, H, S) && none_eq2(F, D, src(&meta, srcX - 1, srcY + 2))) K = M;
                    if all_eq(F, C, H, S) and none_eq(F, D, img_pixel(img, srcX - 1, srcY + 2)):
                        K = M.copy()
                    # if (all_eq3(F, I, B, P) && none_eq2(F, H, src(&meta, srcX - 1, srcY - 2))) M = K;
                    if all_eq(F, I, B, P) and none_eq(F, H, img_pixel(img, srcX - 1, srcY - 2)):
                        M = K.copy()
            # dst
            dstX, dstY = srcX * 2, srcY * 2
            img_scaled[dstY, dstX] = J
            img_scaled[dstY, dstX + 1] = K
            img_scaled[dstY + 1, dstX] = L
            img_scaled[dstY + 1, dstX + 1] = M
            # print(J, K, L, M)
            # break

            A, B, C = B, C, img_pixel(img, srcX + 2, srcY - 1)
            Q, D, E, F, R = D, E, F, R, img_pixel(img, srcX + 3, srcY)
            G, H, I = H, I, img_pixel(img, srcX + 2, srcY + 1)
    return img_scaled