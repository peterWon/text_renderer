#!/usr/env/bin python3
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_img', type=int, default=10, help="Number of images to generate")
    parser.add_argument('--length', type=int, default=10,
                        help='Number of chars in a image, works for chn/random corpus_mode')
    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--img_width', type=int, default=256)

    parser.add_argument('--chars_file', type=str, default='./data/chars/chn.txt')
    parser.add_argument('--config_file', type=str, default='./configs/default.yaml')

    parser.add_argument('--corpus_mode', type=str, default='random', choices=['random', 'chn', 'eng'],
                        help='Different corpus type have different load/get_sample method')
    parser.add_argument('--fonts_dir', type=str, default='./data/fonts/chn')
    parser.add_argument('--bg_dir', type=str, default='./data/bg')
    parser.add_argument('--corpus_dir', type=str, default='./data/corpus')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--tag', type=str, default='default', help='output images are saved under output_dir/{tag} dir')

    parser.add_argument('--debug', action='store_true', default=False, help="output uncroped image")
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--strict', action='store_true', default=False,
                        help="check font supported chars when generating images")
    parser.add_argument('--gpu', action='store_true', default=False, help="use CUDA to generate image")
    parser.add_argument('--num_processes', type=int, default=None,
                        help="Number of processes to generate image. If None, use all cpu cores")

    flags, _ = parser.parse_known_args()
    flags.img_save_dir = os.path.join(flags.output_dir, 'JPEGImages')
    flags.xml_save_dir = os.path.join(flags.output_dir, 'Annotations')

    if os.path.exists(flags.bg_dir):
        num_bg = len(os.listdir(flags.bg_dir))
        flags.num_bg = num_bg

    if not os.path.exists(flags.img_save_dir):
        os.makedirs(flags.img_save_dir)
    if not os.path.exists(flags.xml_save_dir):
        os.makedirs(flags.xml_save_dir)

    return flags
