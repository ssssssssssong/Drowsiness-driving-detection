"""This module provides commonly used image preprocessing functions."""
import cv2
import numpy as np

from mark_operator import MarkOperator

MO = MarkOperator()


def crop_face(image, marks, scale=1.8, shift_ratios=(0, 0)):


    x_min, y_min, _ = np.amin(marks, 0)
    x_max, y_max, _ = np.amax(marks, 0)
    side_length = max((x_max - x_min, y_max - y_min)) * scale


    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2


    img_height, img_width, _ = image.shape
    x_shift, y_shift = np.array(shift_ratios) * side_length

    x_start = int(x_center - side_length / 2 + x_shift)
    y_start = int(y_center - side_length / 2 + y_shift)
    x_end = int(x_center + side_length / 2 + x_shift)
    y_end = int(y_center + side_length / 2 + y_shift)


    border_width = 0
    border_x = min(x_start, y_start)
    border_y = max(x_end - img_width, y_end - img_height)
    if border_x < 0 or border_y > 0:
        border_width = max(abs(border_x), abs(border_y))
        x_start += border_width
        y_start += border_width
        x_end += border_width
        y_end += border_width
        image_with_border = cv2.copyMakeBorder(image, border_width,
                                               border_width,
                                               border_width,
                                               border_width,
                                               cv2.BORDER_CONSTANT,
                                               value=[0, 0, 0])
        image_cropped = image_with_border[y_start:y_end,
                                          x_start:x_end]
    else:
        image_cropped = image[y_start:y_end, x_start:x_end]

    return image_cropped, border_width, (x_start, y_start, x_end, y_end)


def normalize(inputs):

    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Normalization
    return ((inputs / 255.0) - img_mean)/img_std


def rotate_randomly(image, marks, degrees=(-30, 30)):


    degree = np.random.random_sample() * (degrees[1] - degrees[0]) + degrees[0]
    img_height, img_width, _ = image.shape
    rotation_mat = cv2.getRotationMatrix2D(((img_width-1)/2.0,
                                            (img_height-1)/2.0), degree, 1)
    image_rotated = cv2.warpAffine(
        image, rotation_mat, (img_width, img_height))

    marks_rotated = MO.rotate(marks, np.deg2rad(degree),
                              (img_width/2, img_height/2))

    return image_rotated, marks_rotated


def scale_randomly(image, marks, output_size=(256, 256), scale_range=(0, 1)):

    img_height, img_width, _ = image.shape
    face_height, face_width, _ = MO.get_height_width_depth(marks)


    valid_range = min(img_height - face_height, img_width - face_width) / 2


    low, high = (np.array(scale_range) * valid_range).astype(int)
    margin = np.random.randint(low, high)


    x_start = y_start = margin
    x_stop, y_stop = (img_width - margin, img_height - margin)


    image_cropped = image[y_start:y_stop, x_start:x_stop]
    image_resized = cv2.resize(image_cropped, output_size)

    marks -= [margin, margin, 0]
    marks = (marks / (img_width - margin * 2) * output_size[0]).astype(int)

    return image_resized, marks


def flip_randomly(image, marks, probability=0.5):

    if np.random.random_sample() < probability:
        image = cv2.flip(image, 1)
        marks = MO.flip_lr(marks, image.shape[0])

    return image, marks


def generate_heatmaps(marks, img_size, map_size):

    marks_norm = marks / img_size
    heatmaps = MO.generate_heatmaps(marks_norm, map_size=map_size)

    return heatmaps
