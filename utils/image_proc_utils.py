import tensorflow as tf
from tensorflow import keras
from keras import layers


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 127.5 - 1
    input_mask -= 1
    input_mask = tf.cast(input_mask, tf.float32)
    return input_image, input_mask


def normalize_test(input_image):
    input_image = tf.cast(input_image, tf.float32) / 127.5 - 1
    return input_image


def load_image(datapoint, input_shape):
    """
    Used when the dataset is loaded from a directory.
    """
    input_image = tf.image.resize(datapoint['image'], input_shape)
    input_mask = tf.image.resize(datapoint['segmentation_mask'],
                                 input_shape,
                                 # tuple(int(elem/2) for elem in input_shape),
                                 method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def load_image_test(image_path, input_shape):
    """
    Used when the dataset is loaded using tfds.
    """
    data_image = tf.io.read_file(image_path)
    data_image = tf.io.decode_jpeg(data_image, channels=3)
    data_image = tf.image.resize(data_image, input_shape)
    data_image = normalize_test(data_image)
    return data_image


def load_image_alter(image_path, mask_path, input_shape):
    """
    Used when the dataset is loaded using tfds.
    """
    data_image = tf.io.read_file(image_path)
    data_image = tf.io.decode_jpeg(data_image, channels=3)
    data_image = tf.image.resize(data_image, input_shape)

    mask_image = tf.io.read_file(mask_path)
    mask_image = tf.io.decode_png(mask_image, channels=1)
    mask_image = tf.image.resize(mask_image, input_shape, # tuple(int(elem/2) for elem in input_shape),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    data_image, mask_image = normalize(data_image, mask_image)
    return data_image, mask_image


class Augment(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment = tf.keras.layers.RandomFlip(mode="horizontal", seed=20)

    def call(self, inputs, labels):
        # The label data is converted to uint8 so that we can concatenate it with the inputs with datatype float32.
        # labels = tf.image.convert_image_dtype(labels, dtype=tf.float32)
        output = self.augment(layers.concatenate([inputs, labels], axis=-1))
        # This is because the data that I am using has 3 channels for inputs and 1 channel for labels.
        inputs = output[:,:,:,0:3]
        labels = output[:,:,:,3:]
        # Labels converted back to uint8
        # labels = tf.image.convert_image_dtype(labels, dtype=tf.uint8)
        return inputs, labels
