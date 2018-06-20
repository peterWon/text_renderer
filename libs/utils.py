#!/usr/env/bin python3
import glob
import os
import random

import cv2
import matplotlib.pyplot as plt

import numpy as np
import hashlib


def viz_img(text_im, fignum=1):
    """
    text_im : image containing text
    """
    text_im = text_im.astype(int)
    plt.close(fignum)
    plt.figure(fignum)
    plt.imshow(text_im, cmap='gray')
    plt.show(block=True)
    # plt.hold(True)
    #
    # H, W = text_im.shape[:2]
    # plt.gca().set_xlim([0, W - 1])
    # plt.gca().set_ylim([H - 1, 0])
    # plt.show(block=True)


def prob(percent):
    """
    percent: 0 ~ 1, e.g: 如果 percent=0.1，有 10% 的可能性
    """
    assert 0 <= percent <= 1
    if random.uniform(0, 1) <= percent:
        return True
    return False


def draw_box(img, pnts, color):
    """
    :param img: gray image, will be convert to BGR image
    :param pnts: left-top, right-top, right-bottom, left-bottom
    :param color:
    :return:
    """
    if isinstance(pnts, np.ndarray):
        pnts = pnts.astype(np.int32)

    if len(img.shape) > 2:
        dst = img
    else:
        dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    thickness = 1
    linetype = cv2.LINE_AA
    for i in range(len(pnts) // 4): #每一个字符四个点
        cv2.line(dst, (pnts[4*i+0][0], pnts[4*i+0][1]), (pnts[4*i+1][0], pnts[4*i+1][1]), color=color, thickness=thickness,
                 lineType=linetype)
        cv2.line(dst, (pnts[4*i+1][0], pnts[4*i+1][1]), (pnts[4*i+2][0], pnts[4*i+2][1]), color=color, thickness=thickness,
                 lineType=linetype)
        cv2.line(dst, (pnts[4*i+2][0], pnts[4*i+2][1]), (pnts[4*i+3][0], pnts[4*i+3][1]), color=color, thickness=thickness,
                 lineType=linetype)
        cv2.line(dst, (pnts[4*i+3][0], pnts[4*i+3][1]), (pnts[4*i+0][0], pnts[4*i+0][1]), color=color, thickness=thickness,
                 lineType=linetype)
    return dst


def draw_bbox(img, bbox, color):
    pnts = [
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
        [bbox[0], bbox[1] + bbox[3]]
    ]
    return draw_box(img, pnts, color)


def load_bgs(bg_dir):
    dst = []

    for root, sub_folder, file_list in os.walk(bg_dir):
        for file_name in file_list:
            image_path = os.path.join(root, file_name)

            # For load non-ascii image_path on Windows
            bg = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

            dst.append(bg)

    print("Background num: %d" % len(dst))
    return dst


def load_chars(filepath):
    if not os.path.exists(filepath):
        print("Chars file not exists.")
        exit(1)

    ret = ''
    with open(filepath, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            ret += line[0]
    return ret


def md5(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()


if __name__ == '__main__':
    print(md5('test测试'))
