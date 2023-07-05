import tensorflow as tf
import os
import tensorflow_datasets as tfds
import functools
from utils.instantiation_class import *
from utils.image_proc_utils import *

def load_and_split_dataset(BATCH_SIZE, BUFFER_SIZE, load_from_tfds, img_dir, 
                           mask_dir, train_txt_path, val_txt_path, input_shape):
    """
    BATCH_SIZE: (Integer)
        As obvious, the batch size is to be specified.
        
    BUFFER_SIZE: (Integer) 
        As obvious, the buffer size is to be specified.
        
    load_from_tfds: [Boolean] 
        A variable to check if the dataset is 
        loaded using tfds or from a custom pipeline.
        
    img_dir: (String) 
        The path to the directory that contains training and validation images.
        
    mask_dir: (String) 
        The path to the directory that contains the training and 
        validation labelled masks.
        
    train_path_txt: (String) 
        The path to the directory that contains the list of images
        belonging to the training datasets.
        
    val_path_txt: (String) 
        The path to the directory that contains the list of images
        belonging to the validation datasets.
        
    input_shape: (Tuple) 
        The 2-dimensional shape of the input images that will make the dataset. 
    """
    
    if load_from_tfds:
        dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
        train_images = dataset['train']\
                       .map(functools.partial(load_image,
                                              input_shape=input_shape),
                            num_parallel_calls=tf.data.AUTOTUNE)
        validation_images = dataset['test']\
                            .map(functools.partial(load_image,
                                                   input_shape=input_shape),
                                 num_parallel_calls=tf.data.AUTOTUNE)  
        TRAIN_LENGTH = info.splits['train'].num_examples
        VAL_LENGTH = info.splits['test'].num_examples

    else:
        train_data =[]
        train_mask =[]
        with open(train_txt_path) as f:
            for line in f:
                train_data.append(os.path.join(img_dir, 
                                               line.split()[0]+'.jpg'))
                train_mask.append(os.path.join(mask_dir, 
                                               line.split()[0]+'.png'))
        validation_data =[]
        validation_mask =[]
        with open(val_txt_path) as f:
            for line in f:
                validation_data.append(os.path.join(img_dir, 
                                                    line.split()[0]+'.jpg'))
                validation_mask.append(os.path.join(mask_dir,
                                                    line.split()[0]+'.png'))
        train_images = tf.data.Dataset.from_tensor_slices((train_data,
                                                           train_mask))
        validation_images = tf.data.Dataset.from_tensor_slices((validation_data, 
                                                                validation_mask))
        train_images = train_images.map(functools.partial(load_image_alter, 
                                                          input_shape=input_shape),
                                        num_parallel_calls=tf.data.AUTOTUNE)
        validation_images = validation_images.map(functools.partial(load_image_alter, 
                                                                    input_shape=input_shape),
                                                  num_parallel_calls=tf.data.AUTOTUNE)
        TRAIN_LENGTH = len(train_data)
        VAL_LENGTH = len(validation_data)

    train_batches = (train_images
                    # .cache()
                    .shuffle(BUFFER_SIZE)
                    .batch(BATCH_SIZE)
                    .repeat()
                    .map(Augment())
                    .prefetch(buffer_size=tf.data.AUTOTUNE))
    validation_batches = validation_images.batch(BATCH_SIZE)
    return train_batches, validation_batches, TRAIN_LENGTH, VAL_LENGTH