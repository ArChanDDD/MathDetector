import cv2
import numpy as np
from PIL import Image

### CHECK IT BEFORE RUNNING ###
dataset_way = ''
way_to_save_letters = 'partirnated_dataset/'
image_name = '1.png'
letter_size = 80
model_name = 'good_model.hdf5'
letter_indent = 10
###############################

def dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def detector_on_page(page_name):            ### detect letters on page, returns array of latters pics and count of latters
    img = cv2.imread(dataset_way + page_name)
    width = img.shape[1]
    height = img.shape[0]

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    letter_count_in_page = 0
    letters = []
    letters_coords = []

    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in cnts:

        ### criterion for detecting letters ###

        if cv2.contourArea(cnt) > 10:
            (x, y, w, h) = cv2.boundingRect(cnt)

            ### checking intersections ###
            letter_count_in_page += 1
            w = w + min(int(letter_indent / 2), x) + min(int(letter_indent / 2), width - x - w)
            x = x - min(int(letter_indent / 2), x)
            h = h + min(int(letter_indent / 2), y) + min(int(letter_indent / 2), height - y - h)
            y = y - min(int(letter_indent / 2), y)
            s = w * h
            letters_coords.append([x, y, w, h, s])

    letters_coords.sort(key=lambda x: x[4], reverse=True)

    letters_coords = total_check(letters_coords)

    # croping letters
    for i in letters_coords:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]

        cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)

        letter_crop = gray[y:y + h, x:x + w]

        size_max = max(w, h)
        letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
        if w > h:
            y_pos = size_max // 2 - h // 2
            letter_square[y_pos:y_pos + h, 0:w] = letter_crop
        elif w < h:
            x_pos = size_max // 2 - w // 2
            letter_square[0:h, x_pos:x_pos + w] = letter_crop
        else:
            letter_square = letter_crop

        letters.append((x, w, cv2.resize(letter_square, (letter_size, letter_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters, letter_count_in_page


def partirnate_file(image_name):
    letters, k = detector_on_page(image_name)
    print('there are ', len(letters),' letters')
    for i in range(len(letters)):
        img = cv2.cvtColor(letters[i][2],cv2.COLOR_GRAY2RGB)
        #cv2.imshow(str(i), img)
        cv2.imwrite(way_to_save_letters + str(i) + '.jpg', img)
    #cv2.waitKey(0)


def total_check(X):     ### checking for all

    ### checking for intersection. Criterion: square.
    # Дальше смотрим на каждую букву, и проверяем с теми, у которых площадь меньше.
    # Если они достаточно близки, и отношение площади большего прямоугольника к площади внешнего больше 0.8,
    # то мы объединяем их в один.
    obj = 0
    while obj != len(X) - 1:
        x = X[obj][0]
        y = X[obj][1]
        w = X[obj][2]
        h = X[obj][3]
        s = X[obj][4]
        check = True
        while check:
            check = False
            for obj2 in range(obj + 1, len(X)):
                xx = X[obj2][0]
                yy = X[obj2][1]
                ww = X[obj2][2]
                hh = X[obj2][3]

                x_new = min(x, xx)
                y_new = min(y, yy)
                w_new = max(ww, w, x + w - xx, xx + ww - x)
                h_new = max(hh, h, y + h - yy, yy + hh - y)
                if (s/(h_new * w_new) >= 0.7):
                    del X[obj2]
                    X[obj][0] = x_new
                    X[obj][1] = y_new
                    X[obj][2] = w_new
                    X[obj][3] = h_new
                    X[obj][4] = w_new * h_new

                    x = x_new
                    y = y_new
                    w = w_new
                    h = h_new
                    s = w_new * h_new
                    check = True
                    break
        obj += 1
    return X



partirnate_file(image_name)