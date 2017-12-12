#!/usr/bin/python
# -*- coding: latin-1 -*-
from PIL import Image
import os
import matplotlib.pyplot as plt
import scipy.misc as sci
import numpy as np

# Constants
celeba_path = os.getcwd() + '\celebA\img_align_celeba'
out_path = 'celebA\processed'
attr_path = os.getcwd() + '\celebA\Anno\list_attr_celeba.txt'
BLACK = 8
BLOND = 9
BROWN = 11
MALE = 20
YOUNG = 39
attrs = [BLACK, BLOND, BROWN, MALE, YOUNG]




## Preprocess images in database

attr = open(attr_path, 'r')
attr.readline()
attr.readline() # disregard first two lines

celeba_path = celeba_path.decode('iso8859_15')
# out_path = out_path.decode('iso8859_15')
for filename in os.listdir(celeba_path):
    filename = filename.encode('ascii','ignore')

    # Open
    img = Image.open(os.path.join(celeba_path, filename))
    # img = sci.imread(os.path.join(celeba_path, filename))
    # plt.figure()
    # plt.imshow(img)

    # We crop the initial 178x218 size images to 178x178
    img = img.crop((0, 20, 178, 198))
    # img = img[20:198, 0:178, :]
    # plt.figure()
    # plt.imshow(img)

    # Then resize them as 128x128
    img = img.resize((128, 128), Image.ANTIALIAS)
    # img = sci.imresize(img, (128,128))
    # plt.figure()
    # plt.imshow(img)


    # We construct seven domains using the following attributes: hair color (black, blond, brown), gender (male/female), and age (young/old).
    line = attr.readline()
    splits = line.split()
    labels = splits[1:]
    new_labels = []

    for idx, value in enumerate(labels):
        if idx in attrs:
            if int(value) == 1:
                new_labels.append(1)
            else:
                new_labels.append(0)

    labels.append(new_labels)

    # Save modified images
    pic_path = filename.split('.')[0] + '_' + str(new_labels) + '.jpg'
    # pic_path = os.path.join(out_path, pic_path)
    img.save(pic_path)
    break





