
import cv2
import numpy as np
import tensorflow as tf

from fmd.universal import Universal
from preprocessing import (flip_randomly, generate_heatmaps, normalize,
                           rotate_randomly, scale_randomly)


def data_generator(data_dir, name, image_size, number_marks, training):


    # Initialize the dataset with files.
    dataset = Universal(name.decode("utf-8"))
    dataset.populate_dataset(data_dir.decode("utf-8"), key_marks_indices=None)
    dataset.meta.update({"num_marks": number_marks})

    image_size = tuple(image_size)
    width, _ = image_size
    for sample in dataset:
        # Follow the official preprocessing implementation.
        image = sample.read_image("RGB")
        marks = sample.marks

        if training:
            # Rotate the image randomly.
            image, marks = rotate_randomly(image, marks, (-30, 30))

            # Scale the image randomly.
            image, marks = scale_randomly(image, marks, output_size=image_size)

            # Flip the image randomly.
            image, marks = flip_randomly(image, marks)
        else:
            # Scale the image to output size.
            marks = marks / image.shape[0] * width
            image = cv2.resize(image, image_size)

        # Normalize the image.
        image_float = normalize(image.astype(float))

        # Generate heatmaps.
        heatmaps = generate_heatmaps(marks, width, (64, 64))
        heatmaps = np.transpose(heatmaps, (1, 2, 0))

        yield image_float, heatmaps


class WFLWSequence(tf.keras.utils.Sequence):
    """A Sequence implementation for WFLW dataset generation.

    This class is not used in training. It simply demonstrates how to generate
    a TensorFlow dataset by using Keras `Sequence`.
    """

    def __init__(self, data_dir, name, training, batch_size):
        self.training = training
        self.batch_size = batch_size
        self.filenames = []
        self.marks = []


        dataset = Universal(name)
        dataset.populate_dataset(data_dir, key_marks_indices=[
            60, 64, 68, 72, 76, 82])

        for sample in dataset:
            self.filenames.append(sample.image_file)
            self.marks.append(sample.marks)

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_files = self.filenames[index *
                                     self.batch_size:(index + 1) * self.batch_size]
        batch_marks = self.marks[index *
                                 self.batch_size:(index + 1) * self.batch_size]

        batch_x = []
        batch_y = []

        for filename, marks in zip(batch_files, batch_marks):

            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.training:

                image, marks = rotate_randomly(image, marks, (-30, 30))


                image, marks = scale_randomly(image, marks)


                image, marks = flip_randomly(image, marks)
            else:

                marks = marks / image.shape[0] * 256
                image = cv2.resize(image, (256, 256))


            image_float = normalize(image.astype(float))


            _, img_width, _ = image.shape
            heatmaps = generate_heatmaps(marks, img_width, (64, 64))
            heatmaps = np.transpose(heatmaps, (1, 2, 0))


            batch_x.append(image_float)
            batch_y.append(heatmaps)

        return np.array(batch_x), np.array(batch_y)


def build_dataset(data_dir,
                  name,
                  number_marks,
                  image_shape=(256, 256, 3),
                  training=True,
                  batch_size=None,
                  shuffle=True,
                  prefetch=None):

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(image_shape, (64, 64, number_marks)),
        args=[data_dir, name, image_shape[:2], number_marks, training])

    print("Dataset built from generator: {}".format(name))


    if shuffle:
        dataset = dataset.shuffle(1024)


    dataset = dataset.batch(batch_size)

    if prefetch is not None:
        dataset = dataset.prefetch(prefetch)

    return dataset


