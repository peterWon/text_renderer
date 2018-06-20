from PIL import ImageFont, Image, ImageDraw
import random

word = '额' \
       ''
# font_size = random.randint(20, 40)
font = ImageFont.truetype('/home/wz/DeepLearning/ocr/text_renderer/data/fonts/chn/msyh.ttc', 20)
offset = font.getoffset(word)
size = font.getsize(word)
print(offset)
print(size)


with open('/home/wz/DeepLearning/caffe_dir/easy-pvanet/data/VOCdevkit2007/classes_name.txt', 'w') as ofile:
       for i in range(5071):
              ofile.writelines(str(i) + '\n')