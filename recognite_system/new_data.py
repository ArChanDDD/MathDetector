import simple_partirnation as sip
import cv2
import os
import sys
import numpy as np

dataset_way = 'dataset2/dataset/'
way_to_save = 'dataset2/dataset3/'

def print_percentage(p):
    perc = round(p*100, 4)
    sys.stdout.write("\r" + str(perc) + "% done ")
    sys.stdout.flush()

def get_names():
    names = os.listdir('./' + dataset_way)
    names.sort()
    return names

def prepare_dataset(names):
    k = -1
    for name in names:
        k += 1
        print_percentage(k / len(names))
        a, b = sip.detector_on_page(dataset_way + name)
        try:
            img = cv2.resize(a[0][2], (32,32), interpolation=cv2.INTER_AREA)
        except:
            img = cv2.imread(dataset_way + name)
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        cv2.imwrite(way_to_save + str(k) + '.jpg', img)

names = get_names()
prepare_dataset(names)
