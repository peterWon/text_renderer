#!/usr/env/bin python3

"""
Generate training and test images.
"""
import traceback
import numpy as np

import multiprocessing as mp
from itertools import repeat
import os

import cv2

from libs.config import load_config
from libs.timer import Timer
from parse_args import parse_args
import libs.utils as utils
import libs.font_utils as font_utils
from textrenderer.corpus import RandomCorpus, ChnCorpus, EngCorpus
from textrenderer.renderer import Renderer
from tenacity import retry
from libs.pascal_voc_io import *

lock = mp.Lock()
counter = mp.Value('i', 0)
STOP_TOKEN = 'kill'

corpus_classes = {
    "random": RandomCorpus,
    "chn": ChnCorpus,
    "eng": EngCorpus
}

flags = parse_args()
cfg = load_config(flags.config_file)

fonts = font_utils.get_font_paths(flags.fonts_dir)
bgs = utils.load_bgs(flags.bg_dir)

corpus_class = corpus_classes[flags.corpus_mode]
corpus = corpus_class(chars_file=flags.chars_file, corpus_dir=flags.corpus_dir, length=flags.length)
charset = corpus.get_charset()
char2int = dict(zip(charset, range(len(charset))))

renderer = Renderer(corpus, fonts, bgs, cfg,
                    height=flags.img_height,
                    width=flags.img_width,
                    debug=flags.debug,
                    gpu=flags.gpu,
                    strict=flags.strict)


def start_listen(q, fname):
    """ listens for messages on the q, writes to file. """

    f = open(fname, mode='a', encoding='utf-8')
    while 1:
        m = q.get()
        if m == STOP_TOKEN:
            break
        try:
            f.write(str(m) + '\n')
        except:
            traceback.print_exc()

        with lock:
            if counter.value % 1000 == 0:
                f.flush()
    f.close()


@retry
def gen_img_retry(renderer):
    try:
        return renderer.gen_img()
    except Exception:
        print("Retry gen_img")
        raise Exception


def generate_img(img_index, q):
    global flags, lock, counter
    # Make sure different process has different random seed
    np.random.seed()

    im, word, pts = gen_img_retry(renderer)

    base_name = '{:08d}'.format(img_index)

    if not flags.viz:
        fname = os.path.join(flags.img_save_dir, base_name + '.jpg')
        cv2.imwrite(fname, im)

        # write to voc.
        xmlname = os.path.join(flags.xml_save_dir, base_name + '.xml')
        voc_xml = PascalVocWriter(os.path.dirname(fname), base_name + '.jpg', im.shape)
        for index, char in enumerate(word):
            rc = cv2.boundingRect(np.array([pts[4*index], pts[4*index+1], pts[4*index+2],pts[4*index+3]]))
            left, top = rc[0], rc[1]
            right, bot = rc[0] + rc[2], rc[1] + rc[3]

            # print(char, left, top, right, bot)
            voc_xml.addBndBox(int(left), int(top), int(right), int(bot), char)#str(char2int[char])
        voc_xml.save(xmlname)

        label = "{} {}".format(base_name, word)
        q.put(label)

        with lock:
            counter.value += 1
            print_end = '\n' if counter.value == flags.num_img else '\r'
            if counter.value % 100 == 0 or counter.value == flags.num_img:
                print("{}/{} {:2d}%".format(counter.value,
                                            flags.num_img,
                                            int(counter.value / flags.num_img * 100)),
                      end=print_end)
    else:
        utils.viz_img(im)


def sort_labels(tmp_label_fname, label_fname):
    lines = []
    with open(tmp_label_fname, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = sorted(lines)
    with open(label_fname, mode='w', encoding='utf-8') as f:
        for line in lines:
            f.write(line[9:])


def restore_exist_labels(save_dir, label_path):
    # 如果目标目录存在 labels.txt 则向该目录中追加图片
    start_index = 0
    if os.path.exists(label_path):
        print('Generate more text images in %s' % flags.output_dir)
        start_index = len(utils.load_chars(label_path))
    else:
        print('Generate text images in %s' % flags.output_dir)
    return start_index

def write_labelmap(fpath, char2int):
    with open(fpath, 'w') as ofile:
        for k, val in char2int.items():
            ofile.writelines(k + '\n')

def write_freq(fpath, charset_freq):
    with open(fpath, 'w') as ofile:
        for k, val in charset_freq.items():
            ofile.writelines(k+' '+str(val)+'\n')

if __name__ == "__main__":
    if flags.viz == 1:
        flags.num_processes = 1

    tmp_label_path = os.path.join(flags.output_dir, 'tmp_labels.txt')
    label_path = os.path.join(flags.output_dir, 'labels.txt')

    manager = mp.Manager()
    q = manager.Queue()

    start_index = restore_exist_labels(flags.output_dir, label_path)

    timer = Timer(Timer.SECOND)
    timer.start()
    with mp.Pool(processes=flags.num_processes) as pool:
        if not flags.viz:
            pool.apply_async(start_listen, (q, tmp_label_path))

        pool.starmap(generate_img, zip(range(start_index, start_index + flags.num_img), repeat(q)))

        q.put(STOP_TOKEN)
        pool.close()
        pool.join()
    timer.end("Finish generate data")

    if not flags.viz:
        sort_labels(tmp_label_path, label_path)

    write_labelmap(fpath = os.path.join(flags.output_dir, '..', 'classes_name.txt'), char2int = char2int)
    write_freq(fpath = os.path.join(flags.output_dir, '..', 'frequency.txt'), charset_freq = renderer.corpus.get_frequency())