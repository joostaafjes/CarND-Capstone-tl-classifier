import os
import time
from shutil import copyfile

from PIL import ImageDraw, ImageFont
from keras.preprocessing.image import load_img
from pathlib2 import Path

from tl_classifier import TLClassifier, load_image_into_numpy_array
import tensorflow as tf

tl_classifier = TLClassifier()
tf_colors = ['red', 'yellow', 'green', None, 'unknown']
output_base_path='./output_images/'

for color in tf_colors:
    if color:
        dirname = os.path.dirname(output_base_path + color + '/')
        path = Path(dirname)
        path.mkdir(exist_ok=True, parents=True)

cnt_error = 0
cnt_ok = 0

idx = 0
for root, dirs, files in os.walk("images", topdown=False):
    for filename in files:
        if filename.startswith('.DS_Store'):
            continue
        path = root + '/' + filename
        print('start processing...{}'.format(path))
        image = load_img(root + '/' + filename)  # this is a PIL image
        image_np = load_image_into_numpy_array(image)
        start = time.time()
        color, _ = tl_classifier.get_classification(image_np, idx)
        elapsed = time.time() - start
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 55)
        print('elapsed time:', elapsed, ' s')
        print(path, tf_colors[color])
        # output path either base dir or subdir for color
        # output_path = output_base_path + filename
        output_path = output_base_path + tf_colors[color] + '/' + filename
        # draw text and write file in main output dir
        draw.text((10, 10),
                  tf_colors[color],
                  font=font,
                  fill=(255, 255, 0),)
        copyfile(root + '/' + filename, output_path)
        image.save(output_path)

        if path.lower().find(tf_colors[color]) == -1:
            cnt_error += 1
        else:
            cnt_ok += 1
        # tmp
        # if cnt_error + cnt_ok > 5:
        #     break
        idx += 1
    # if cnt_error + cnt_ok > 5:
    #     break

print('Succes rate:', 100 * cnt_ok / (cnt_ok + cnt_error), ' %')

