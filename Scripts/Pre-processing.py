##

import cv2, numpy, os
import pandas as pd
from threading import Thread
import numpy as np
import glob

##

number_of_threads = 15
preprocessing = "denoising2"
overwrite = True

##

db_folder_kaggle = os.path.join('..', "Database")
images_folder_kaggle = os.path.join(db_folder_kaggle, "images")
preprocessed_folder_kaggle = os.path.join(db_folder_kaggle, "preprocessing images", preprocessing)
labels_dir = os.path.join(db_folder_kaggle, "labels")
labels = pd.read_csv(os.path.join(labels_dir, "labels.csv"))


##


def detect_xyr(image, min_radius_ratio=.33, max_radius_ratio=.6):
    width = image.shape[1]
    height = image.shape[0]
    myMinWidthHeight = min(width, height)
    myMinRadius = round(myMinWidthHeight * min_radius_ratio)
    myMaxRadius = round(myMinWidthHeight * max_radius_ratio)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450,
                               param1=120, param2=32,
                               minRadius=myMinRadius,
                               maxRadius=myMaxRadius)
    (x, y, r) = (0, 0, 0)
    found_circle = False
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if (circles is not None) and (len(circles == 1)):
            x1, y1, r1 = circles[0]
            if (2 / 5 * width) < x1 < (3 / 5 * width) and (2 / 5 * height) < y1 < (3 / 5 * height):
                x, y, r = circles[0]
                found_circle = True
    if not found_circle:
        x = image.shape[1] // 2
        y = image.shape[0] // 2
        temp_x = image[int(image.shape[0] / 2), :, :].sum(1)
        r = int((temp_x > temp_x.mean() / 12).sum() / 2)
    return (x, y, r)


def center_eye_bulb(image, x, y, r):
    (image_height, image_width) = (image.shape[0], image.shape[1])
    image_left = int(max(0, x - r))
    image_right = int(min(x + r, image_width - 1))
    image_top = int(max(0, y - r))
    image_bottom = int(min(y + r, image_height - 1))
    image = image[image_top: image_bottom, image_left:image_right, :]
    return image


def set_eyebulb_diameter(image, diameter):
    if image.shape[1] > image.shape[0]:
        image = cv2.resize(image, (diameter, int(diameter * image.shape[0] / image.shape[1])))
    else:
        image = cv2.resize(image, (int(diameter * image.shape[1] / image.shape[0]), diameter))
    return image


def substract_local_average(image, filter_size, diameter):
    b = np.zeros(image.shape)
    cv2.circle(b, (image.shape[1] // 2, image.shape[0] // 2), int(diameter // 2 * 0.9), (1, 1, 1), -1, 8, 0)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), filter_size), -4, 128) * b + 128 * (1 - b)
    return image


def crop(image, x, y, r, percentage):
    r = int(percentage * r)
    height, width, _ = image.shape
    mask = np.zeros((height, width, 3), np.uint8)
    mask = cv2.circle(mask, (x, y), r, (1, 1, 1), -1)
    return np.multiply(image, mask)


def add_padding(image):
    (image_height, image_width) = (image.shape[0], image.shape[1])
    left_padding = 0 if image_width > image_height else (image_height - image_width) // 2
    right_padding = 0 if image_width > image_height else (image_height - image_width) // 2
    top_padding = 0 if image_width < image_height else (image_width - image_height) // 2
    bottom_padding = 0 if image_width < image_height else int(image_width - image_height) // 2
    image = cv2.copyMakeBorder(image, top=top_padding, bottom=bottom_padding, left=left_padding, right=right_padding,
                               borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image


def adaptation_preprocess(image, desired_size=540, diameter=540, filter_size=9, percentage=.9, remove=30, **kwargs):
    assert 2 * remove <= desired_size
    (x, y, r) = detect_xyr(image.copy())
    image = center_eye_bulb(image, x, y, r)
    image = set_eyebulb_diameter(image, diameter)
    image = substract_local_average(image, filter_size, diameter)
    image = crop(image, image.shape[1] // 2, image.shape[0] // 2, max(image.shape[1] // 2, image.shape[0] // 2),
                 percentage)
    image = add_padding(image)
    image = cv2.resize(image, (desired_size, desired_size), interpolation=cv2.INTER_NEAREST)
    return image


##


def original_preprocessing(img, scale=270, **kwargs):
    x = img[img.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    img = cv2.resize(img, (0, 0), fx=s, fy=s)
    b = numpy.zeros(img.shape)
    cv2.circle(b, (img.shape[1] // 2, img.shape[0] // 2), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), scale / 30), -4, 128) * b + 128 * (1 - b)
    return img


##

def center_preprocessing(image, desired_size=540, diameter=540, percentage=.9, remove=30, **kwargs):
    assert 2 * remove <= desired_size
    (x, y, r) = detect_xyr(image.copy())
    image = center_eye_bulb(image, x, y, r)
    image = set_eyebulb_diameter(image, diameter)
    image = crop(image, image.shape[1] // 2, image.shape[0] // 2, max(image.shape[1] // 2, image.shape[0] // 2),
                 percentage)
    image = add_padding(image)
    image = cv2.resize(image, (desired_size, desired_size), interpolation=cv2.INTER_NEAREST)
    return image


##

def adaptation_denoising_preprocess(image, desired_size=540, diameter=540, filter_size=9, percentage=.9, remove=30,
                                    **kwargs):
    assert 2 * remove <= desired_size
    (x, y, r) = detect_xyr(image.copy())
    image = center_eye_bulb(image, x, y, r)
    image = set_eyebulb_diameter(image, diameter)
    image = cv2.fastNlMeansDenoisingColored(image.astype("uint8"), None, hColor=kwargs.get("h") if kwargs.get("h") else 3)
    image = substract_local_average(image, filter_size, diameter)
    image = crop(image, image.shape[1] // 2, image.shape[0] // 2, max(image.shape[1] // 2, image.shape[0] // 2),
                 percentage)
    image = add_padding(image)
    image = cv2.resize(image, (desired_size, desired_size), interpolation=cv2.INTER_NEAREST)
    return image


##

def denoising_preprocess(image, desired_size=540, diameter=540, percentage=.9, remove=30, **kwargs):
    assert 2 * remove <= desired_size
    (x, y, r) = detect_xyr(image.copy())
    image = center_eye_bulb(image, x, y, r)
    image = set_eyebulb_diameter(image, diameter)
    image = cv2.fastNlMeansDenoisingColored(image.astype("uint8"), None, hColor=kwargs.get("h") if kwargs.get("h") else 3)
    image = crop(image, image.shape[1] // 2, image.shape[0] // 2, max(image.shape[1] // 2, image.shape[0] // 2),
                 percentage)
    image = add_padding(image)
    image = cv2.resize(image, (desired_size, desired_size), interpolation=cv2.INTER_NEAREST)
    return image


##


def apply_preprocessing(labels, number_threads, assigned):
    chosen_labels = labels.iloc[list(range(assigned, len(labels), number_threads))].copy()
    kargs = {}
    if preprocessing == "adaptation":
        func = adaptation_preprocess
    elif preprocessing == "original":
        func = original_preprocessing
    elif preprocessing == "center":
        func = center_preprocessing
    elif preprocessing.startswith("adaptationDenoising"):
        func = adaptation_denoising_preprocess
        kargs["h"] = int(preprocessing[len("adaptationDenoising"):])
    elif preprocessing.startswith("denoising"):
        func = denoising_preprocess
        kargs["h"] = int(preprocessing[len("denoising"):])
    else:
        func = adaptation_preprocess
    for i in range(len(chosen_labels)):
        image_data = chosen_labels.iloc[i]
        image_path = os.path.join(images_folder_kaggle, image_data['image'] + ".jpeg")
        new_path = os.path.join(preprocessed_folder_kaggle, image_data['image'] + ".jpeg")
        if overwrite or not os.path.exists(new_path):
            try:
                img = cv2.imread(image_path)
                img = func(img, **kargs)
                cv2.imwrite(new_path, img)
            except Exception as e:
                print(image_data["image"])


threads = []

print("Inicio preprocesamiento")

for i in range(number_of_threads):
    t = Thread(target=apply_preprocessing, args=(labels, number_of_threads, i))
    threads.append(t)
    t.start()
    print(F"Empezó hilo {i}")

for i in range(number_of_threads):
    threads[i].join()

print("Acabo preprocesamiento")
