import sys, os ,time
import math, shutil
from PIL import Image, ExifTags


def fix(numImgs):
    classes = os.listdir('letter_dataset')
    imgPerClass = math.floor(numImgs/len(classes))

    if f'letter_dataset_{numImgs}' not in os.listdir():
        os.mkdir(f'letter_dataset_{numImgs}')

    for c in classes:
        newImgs = os.listdir(f'letter_dataset/{c}')[:imgPerClass]
        if c not in os.listdir(f'letter_dataset_{numImgs}'):
            os.mkdir(f'letter_dataset_{numImgs}/{c}')
        for img in newImgs:
            shutil.copy(f'letter_dataset/{c}/{img}', f'letter_dataset_{numImgs}/{c}/{img}')

fix(5000)

