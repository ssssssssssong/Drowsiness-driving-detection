import argparse
import logging
import traceback
import os

import cv2
import numpy as np
from tqdm import tqdm

import fmd
from mark_guardians import check_mark_location
from mark_operator import MarkOperator
from mesh_detector import MeshDetector
from mesh_record_operator import MeshRecordOperator

# Get the command line argument.
parser = argparse.ArgumentParser()
#parser.add_argument("--loglevel", type=str, default="info",
#                    help="The logging level.")
parser.add_argument("--loglevel", type=str, default="DEBUG",
                    help="The logging level.")
args = parser.parse_args()


def setup_logger():

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))


    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    console_hdlr = logging.StreamHandler()
    console_hdlr.setFormatter(log_formatter)


    file_hdlr = logging.FileHandler('data_generation.log')
    file_hdlr.setFormatter(log_formatter)


    logger = logging.getLogger(__name__)
    logger.addHandler(console_hdlr)
    logger.addHandler(file_hdlr)
    logger.setLevel(numeric_level)
    return logger



logger = setup_logger()


def process(dataset, index_start_from=0):

    logger.info("Starting to process dataset: {}".format(dataset.meta['name']))

    # Keep a record of the current location.
    current_sample_index = -1

    # Count the samples considered invalid.
    num_invalid_samples = 0

    # Construct a mark operator to transform the marks.
    mo = MarkOperator()

    # Construct a face mesh detector to generate face mesh from image.
    md = MeshDetector("assets/face_landmark.tflite")

    # Construct a writer for TFRecord files.
    tf_writer = MeshRecordOperator(
        "tfrecord/{}.record".format(dataset.meta["name"]))


    try:
        # Enumerate all the samples in dataset.
        for sample in tqdm(dataset):

            current_sample_index += 1
            if current_sample_index < index_start_from:
                continue


            image = sample.read_image()
            marks = sample.marks


            image_translated, trans_vector = move_face_to_center(
                image, marks, mo)
            marks_translated = marks[:, :2] + trans_vector

            # Second, align the face. This happens in the 2D space.
            image_rotated, degrees = rotate_to_vertical(
                image_translated, sample, mo)
            img_height, img_width, _ = image.shape
            marks_rotated = mo.rotate(
                marks_translated, degrees/180*np.pi, (img_width/2, img_height/2))

            # Third, try to crop the face area out. Pad the image if necessary.
            image_cropped, padding, bbox = crop_face(
                image_rotated, marks, scale=1.7)
            mark_cropped = marks_rotated + \
                padding - np.array([bbox[0], bbox[1]])

            # Last, resize the face area.
            image_resized = cv2.resize(image_cropped, (384, 384))
            mark_resized = mark_cropped * (384 / image_cropped.shape[0])

            # Show all the image processed in debug mode.
            if args.loglevel.upper() == "DEBUG":
                esc_key = show_debug_images(image, marks,
                                            image_translated, marks_translated,
                                            image_rotated, marks_rotated,
                                            image_resized, mark_resized,
                                            padding, bbox)
                if esc_key:
                    break

            # Now the cropped face and marks are available, do whatever you want.

            # Save the processed image and new marks to files.
            export_dir = "wflw_cropped/test"

            img_file = os.path.join(
                export_dir, "{}_{}".format(current_sample_index, os.path.basename(sample.image_file)))
            mark_file = img_file.split(".")[-2] + ".json"

            cv2.imwrite(img_file, image_resized)
            sample.marks = mark_resized
            sample.save_mark_to_json(mark_file)

    except Exception:
        logger.error(
            "Error {}. sample index: {}".format(traceback.format_exc(), current_sample_index))
    finally:
        # Summary
        logger.info("Dataset done. Processed samples: {}, invalid samples: {}".format(
            current_sample_index+1, num_invalid_samples))


def move_face_to_center(image, marks, mo):

    img_height, img_width, _ = image.shape
    face_center = mo.get_center(marks)[:2]
    translation_mat = np.array([[1, 0, img_width / 2 - face_center[0]],
                                [0, 1, img_height / 2 - face_center[1]]])
    image_translated = cv2.warpAffine(
        image, translation_mat, (img_width, img_height))

    translation_vector = np.array(
        [img_width / 2 - face_center[0],  img_height / 2 - face_center[1]])

    return image_translated, translation_vector


def rotate_to_vertical(image, sample, mo):

    img_height, img_width, _ = image.shape
    key_marks = sample.get_key_marks()[:, :2]
    vector_eye = (key_marks[3] - key_marks[0])
    degrees = mo.get_angle(vector_eye, np.array([100, 0]))
    rotation_mat = cv2.getRotationMatrix2D(
        ((img_width-1)/2.0, (img_height-1)/2.0), -degrees, 1)
    image_rotated = cv2.warpAffine(
        image, rotation_mat, (img_width, img_height))

    return image_rotated, degrees


def crop_face(image, marks, scale=1.8, shift_ratios=(0, 0)):

    # How large the bounding box is?
    x_min, y_min, _ = np.amin(marks, 0)
    x_max, y_max, _ = np.amax(marks, 0)
    side_length = max((x_max - x_min, y_max - y_min)) * scale

    # Face box is scaled, get the new corners, shifted.
    img_height, img_width, _ = image.shape
    x_shift, y_shift = np.array(shift_ratios) * side_length

    x_start = int(img_width / 2 - side_length / 2 + x_shift)
    y_start = int(img_height / 2 - side_length / 2 + y_shift)
    x_end = int(img_width / 2 + side_length / 2 + x_shift)
    y_end = int(img_height / 2 + side_length / 2 + y_shift)

    # In case the new bbox is out of image bounding.
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


def show_debug_images(image, marks,
                      image_translated, marks_translated,
                      image_rotated, marks_rotated,
                      image_resized, mark_resized,
                      padding, bbox):
    # Resconstruct the padded image.
    if padding != 0:
        image_padded = cv2.copyMakeBorder(image_rotated, padding,
                                          padding,
                                          padding,
                                          padding,
                                          cv2.BORDER_CONSTANT,
                                          value=[0, 0, 0])
        cv2.rectangle(image_padded, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (255, 255, 255), 3)
        cv2.imshow("Padded", image_padded)
    else:
        cv2.rectangle(image_rotated, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (255, 255, 255), 3)

    # Draw original marks.
    image_original = image.copy()
    fmd.mark_dataset.util.draw_marks(image_original, marks, 1)

    # Draw translated marks.
    fmd.mark_dataset.util.draw_marks(image_translated, marks_translated, 1)

    # Draw rotated marks.
    fmd.mark_dataset.util.draw_marks(image_rotated, marks_rotated, 1)

    # Draw resized marks.
    fmd.mark_dataset.util.draw_marks(image_resized, mark_resized, 1)

    # Show them all, in an uniform manner.
    height = 512
    width = int(image.shape[1] * height / image.shape[0])

    cv2.imshow("Original", cv2.resize(image_original, (width, height)))
    cv2.imshow("Translated", cv2.resize(image_translated, (width, height)))
    cv2.imshow("Rotated", cv2.resize(image_rotated, (width, height)))
    cv2.imshow("Face sample", image_resized)

    return cv2.waitKey() == 27


if __name__ == "__main__":
    # Set the dataset directory you are going to use.

    wflw_dir = "wflw/WFLW_images"


    # Construct the datasets.


    # WFLW
    ds_wflw = fmd.wflw.WFLW(False, "wflw_train")
    ds_wflw.populate_dataset(wflw_dir)
    process(ds_wflw)


