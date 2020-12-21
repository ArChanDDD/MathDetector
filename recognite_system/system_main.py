import system_partirnation as syp
import simple_partirnation as sip
import Test_model as TM
import cv2
import glob
import os
import numpy as np
from keras.models import load_model

dataset_way = '4test/'
way_to_save_letters = 'test_data/'
image_name = 'system2.png'
letter_size = 80
model_name = 'new_model_1.hdf5'
letter_indent = 10
img = cv2.imread(dataset_way + image_name)
pict_width = img.shape[0]
pict_height = img.shape[1]
letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', '*', 'b', 'c', ':', ',', '=', 'f', 'g', 'k',
           '(', '-', '+', ')', '/', 'x', 'y', 'z']



def recognite(img, w, h, model):
    img = img / 255.0
    img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
    ans = model.predict_classes(img)


def from_gray_to_color(images):
    mm = []
    for i in range(len(images)):
        mas = images[i][1]
        m = np.zeros((mas.shape[0], mas.shape[1], 3))
        for i in range(mas.shape[0]):
            for j in range(mas.shape[1]):
                m[i,j] = [mas[i, j], mas[i, j], mas[i, j]]
                m = m.astype('int32')
        mm.append(m)
    return mm


before, after = syp.partirnate_file(dataset_way + image_name)
model = load_model(model_name)

file = open('output.txt', 'w')
file.write('$$\n')

for i in range(len(before)):
    img = before[i][1]
    cv2.imwrite(dataset_way + 'before.jpg', img)
    #cv2.imshow('1', img)
    sip.partirnate_file(dataset_way + 'before.jpg')
    letter = TM.Test_model(model, 32, 32)
    for j in letter:
        file.write(letters[j])
    file.write(' \\\ \n')
    #cv2.waitKey(0)

files = glob.glob('partirnated_dataset/*')
for f in files:
    os.remove(f)


file.write('\\begin{cases}\n')
for i in range(len(after)):
    files = glob.glob('partirnated_dataset/*')
    for f in files:
        os.remove(f)
    img = after[i][1]
    #cv2.imshow('1', img)
    cv2.imwrite(dataset_way + 'after.jpg', img)
    sip.partirnate_file(dataset_way + 'after.jpg')
    letter = TM.Test_model(model, 32, 32)
    for j in letter:
        file.write(letters[j])
    file.write(' \\\ \n')
    #cv2.waitKey(0)
file.write('\\end{cases}\n')

file.write('$$\n')
file.close()



