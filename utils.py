from PIL import Image
import numpy as np


def get_training_imgs(path, scale_factor=4/3):
    img = np.array(Image.open(path), dtype=np.uint8)
    h, w = img.shape[0], img.shape[1]
    if h > w:
        w_ = 250 * w // h
        max_img = np.array(Image.fromarray(img).resize([w_, 250]), dtype=np.uint8)
    else:
        h_ = 250 * h // w
        max_img = np.array(Image.fromarray(img).resize([250, h_]), dtype=np.uint8)
    if h < w:
        w_ = 25 * w // h
        min_img = np.array(Image.fromarray(img).resize([w_, 25]), dtype=np.uint8)
    else:
        h_ = 25 * h // w
        min_img = np.array(Image.fromarray(img).resize([25, h_]), dtype=np.uint8)
    min_h, min_w = min_img.shape[0], min_img.shape[1]
    max_h, max_w = max_img.shape[0], max_img.shape[1]
    heights = []
    width = []
    max_hw = min_h if min_h > min_w else min_w
    n = 0
    while max_hw * scale_factor ** n < 250:
        heights.append(min_h * scale_factor ** n)
        width.append(min_w * scale_factor ** n)
        n += 1
    heights.append(max_h)
    width.append(max_w)
    imgs = []
    for h, w in zip(heights, width):
        imgs.append(np.array(Image.fromarray(img).resize([int(w), int(h)]), dtype=np.uint8))
    return imgs

# imgs = get_training_imgs("D:/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000046.jpg")
# for img in imgs:
#     Image.fromarray(img).show()
# a = 0