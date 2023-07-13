#!/usr/bin/env python

# use the tensorboard import if you wish to open the tensorboard log files after training.
# import tensorboard
import getopt
import sys
from model_and_training.model_utilfuncs import *
from model_and_training.model_instances import *
from model_and_training.train import *
from utils.instantiation_class import *
from utils.image_proc_utils import *
from utils.load_and_split_dataset import *
from utils.display_utils import *
from config_and_params import *


def main_train(model_num, EPOCHS, learn_r):
    (train_batches,
    validation_batches, 
    TRAIN_LENGTH, 
    VAL_LENGTH) = load_and_split_dataset(
        BATCH_SIZE, 
        BUFFER_SIZE, 
        load_from_tfds, 
        img_dir, 
        mask_dir, 
        train_txt_path, 
        val_txt_path, 
        input_shape[:-1]
        )
    samples = []
    for images, masks in train_batches.take(3):
        samples.append(images[0])
        samples.append(masks[0])
    display('Dataset Sample', 'no_models', samples, 3,
            cwd, mode=0)
    if model_num == 0:
        model = UNet_model(train_batches, validation_batches, input_shape, output_classes, 
                            cwd, BATCH_SIZE, TRAIN_LENGTH, VAL_LENGTH, 
                            VAL_SUBSPLITS, EPOCHS)
    elif model_num == 1:
        model = PixelDenseNet(train_batches, validation_batches, input_shape, output_classes,
                                cwd, BATCH_SIZE, TRAIN_LENGTH, VAL_LENGTH, 
                                VAL_SUBSPLITS, EPOCHS)
    elif model_num == 2:
        model = iPixelDenseNet(train_batches, validation_batches, input_shape, output_classes, 
                                cwd, BATCH_SIZE, TRAIN_LENGTH, VAL_LENGTH, 
                                VAL_SUBSPLITS, EPOCHS)
    elif model_num == 3:
        model = modifiediPixelDenseNet(train_batches, validation_batches, input_shape, output_classes, 
                                        cwd, BATCH_SIZE, TRAIN_LENGTH, VAL_LENGTH, 
                                        VAL_SUBSPLITS, EPOCHS)
    else: # control model, to test.
        model = Deeplab(train_batches, validation_batches, input_shape, output_classes, 
                        cwd, BATCH_SIZE, TRAIN_LENGTH, VAL_LENGTH, 
                        VAL_SUBSPLITS, EPOCHS)
    model(cwd, learn_r, samples)


if __name__ == '__main__':
    argumentList = sys.argv[1:]   
    # Options
    options = "m:e:"  
    # Long options
    long_options = ["model_num=, epochs="]
    
    try:
        # default values if no argument is passed.
        model_num = 0
        EPOCHS = 1
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        # checking each argument
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-m", "--model_num"):
                model_num = int(currentValue)
            elif currentArgument in ("-e", "--epochs"):
                EPOCHS = int(currentValue)
        learn_r = 1e-3
        main_train(model_num, EPOCHS, learn_r)
                
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))