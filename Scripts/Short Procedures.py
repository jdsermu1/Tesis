##

import os
import numpy as np
import pandas as pd
import cv2
import glob
import random
from matplotlib import pyplot as plt

##
# Join trainLabels.csv and testLabels.csv and create labels.csv. All annotation should appear in this file even if some
# images are damaged and preprocessing operations do not apply to them, in case this happens this should be noted in the
# file corresponding the preprocessing made.

# labels_folder = os.path.join("Database", "labels")
#
# df_train_labels = pd.read_csv(os.path.join(labels_folder, "trainLabels.csv"))
# df_test_labels = pd.read_csv(os.path.join(labels_folder, "testLabels.csv"))
#
# df_labels = pd.concat([df_test_labels.drop(columns=["Usage"]), df_train_labels])
#
# print(f'Todas los nombres de imagenes son unicos: {len(pd.unique(df_labels["image"])) == len(df_labels)}')
#
# df_labels.to_csv(os.path.join(labels_folder, "labels.csv"), index=False)

##
# Find out distribution of labels in classes

# labels = pd.read_csv(os.path.join("Database", "labels", "labels.csv"))
#
# stats = pd.DataFrame({"percentage": labels["level"].value_counts(normalize=True)*100,
#                       "count": labels["level"].value_counts()})
#
# stats


##
# Compare results Li 2019 get recall and precision
#

# matrix_nn1 = np.array([[39031, 2693, 2361, 75, 160],
#                        [118, 438, 185, 0, 1],
#                        [339, 626, 5058, 738, 392],
#                        [0, 0, 170, 332, 102],
#                        [43, 5, 85, 69, 551]])
#
# recall_nn1 = np.diagonal(np.divide(matrix_nn1, matrix_nn1.sum(0, keepdims=True)))
#
# precision_nn1 = np.diagonal(np.divide(matrix_nn1, matrix_nn1.sum(1, keepdims=True)))
#
# print("Precision", precision_nn1, "Recall", recall_nn1)

##

# labels = pd.read_csv(os.path.join("Database", "labels", "labelsPreprocessingOriginalStrategy4.csv"))
# labels = labels[labels["set"] == "train"]
#
# stats = pd.DataFrame({"percentage": labels["level"].value_counts(normalize=True)*100,
#                       "count": labels["level"].value_counts()})
#
# stats

##
#
# def detect_xyr(image, min_radius_ratio=.33, max_radius_ratio=.6):
#     width = image.shape[1]
#     height = image.shape[0]
#     myMinWidthHeight = min(width, height)
#     myMinRadius = round(myMinWidthHeight * min_radius_ratio)
#     myMaxRadius = round(myMinWidthHeight * max_radius_ratio)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450,
#                                param1=120, param2=32,
#                                minRadius=myMinRadius,
#                                maxRadius=myMaxRadius)
#     (x, y, r) = (0, 0, 0)
#     found_circle = False
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         if (circles is not None) and (len(circles == 1)):
#             x1, y1, r1 = circles[0]
#             if (2 / 5 * width) < x1 < (3 / 5 * width) and (2 / 5 * height) < y1 < (3 / 5 * height):
#                 x, y, r = circles[0]
#                 found_circle = True
#     if not found_circle:
#         x = image.shape[1] // 2
#         y = image.shape[0] // 2
#         temp_x = image[int(image.shape[0] / 2), :, :].sum(1)
#         r = int((temp_x > temp_x.mean() / 12).sum() / 2)
#     return (x, y, r)
#
#
# def center_eye_bulb(image, x, y, r):
#     (image_height, image_width) = (image.shape[0], image.shape[1])
#     image_left = int(max(0, x - r ))
#     image_right = int(min(x + r, image_width - 1))
#     image_top = int(max(0, y - r))
#     image_bottom = int(min(y + r, image_height - 1))
#     image= image[image_top: image_bottom, image_left:image_right, :]
#     return image
#
#
# def set_eyebulb_diameter(image, diameter):
#     if image.shape[1]>image.shape[0]:
#         image = cv2.resize(image, (diameter, int(diameter*image.shape[0]/image.shape[1])))
#     else:
#         image = cv2.resize(image, (int(diameter*image.shape[1]/image.shape[0]), diameter))
#     return image
#
#
# def substract_local_average(image, filter_size, diameter):
#     b = np.zeros(image.shape)
#     cv2.circle(b, (image.shape[1]//2,image.shape[0]//2),int(diameter//2*0.9),(1,1,1),-1,8,0)
#     image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), filter_size),-4, 128)*b+128*(1-b)
#     return image
#
#
# def crop(image, x, y, r, percentage):
#     r = int(percentage*r)
#     height, width, _ = image.shape
#     mask = np.zeros((height, width, 3), np.uint8)
#     mask = cv2.circle(mask, (x,y), r, (1,1,1), -1)
#     return np.multiply(image, mask)
#
#
# def add_padding(image):
#     (image_height, image_width) = (image.shape[0], image.shape[1])
#     left_padding = 0 if image_width>image_height else (image_height-image_width)//2
#     right_padding = 0 if image_width>image_height else (image_height-image_width)//2
#     top_padding = 0 if image_width<image_height else (image_width-image_height)//2
#     bottom_padding = 0 if image_width<image_height else int(image_width-image_height)//2
#     image = cv2.copyMakeBorder(image,top=top_padding,bottom=bottom_padding, left=left_padding,right=right_padding,
#                                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
#     return image
#
#
# def adaptation_preprocess(image, desired_size=540, diameter=540, filter_size=9, percentage=.9, remove=30):
#     assert 2*remove<=desired_size
#     (x, y, r) = detect_xyr(image.copy())
#     image = center_eye_bulb(image, x, y, r)
#     image = set_eyebulb_diameter(image, diameter)
#     image = substract_local_average(image, filter_size, diameter)
#     image = crop(image, image.shape[1]//2, image.shape[0]//2, max(image.shape[1]//2, image.shape[0]//2), percentage)
#     image = add_padding(image)
#     image = cv2.resize(image, (desired_size, desired_size), interpolation=cv2.INTER_NEAREST)
#     return image
#
#
# def adaptation_preprocess2(image, desired_size=540, diameter=540, filter_size=9, percentage=.9, remove=30):
#     assert 2*remove<=desired_size
#     (x, y, r) = detect_xyr(image.copy())
#     image = center_eye_bulb(image, x, y, r)
#     image = set_eyebulb_diameter(image, diameter)
#     image = cv2.fastNlMeansDenoisingColored(image.astype("uint8"), None)
#     image = substract_local_average(image, filter_size, diameter)
#     image = crop(image, image.shape[1]//2, image.shape[0]//2, max(image.shape[1]//2, image.shape[0]//2), percentage)
#     image = add_padding(image)
#     image = cv2.resize(image, (desired_size, desired_size), interpolation=cv2.INTER_NEAREST)
#     return image
#
#
#
# image_name = random.choice(glob.glob(os.path.join("../Database/images/*.jpeg")))
#
# img = cv2.imread(image_name)
#
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
#
# img_adaptation = adaptation_preprocess(img)
#
# # print(type(img_adaptation))
# plt.imshow(cv2.cvtColor(img_adaptation.astype("uint8"), cv2.COLOR_BGR2RGB))
# plt.show()
#
# img_adaptation = adaptation_preprocess2(img)
#
# plt.imshow(cv2.cvtColor(img_adaptation.astype("uint8"), cv2.COLOR_BGR2RGB))
# plt.show()


