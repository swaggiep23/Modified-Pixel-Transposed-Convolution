#!/usr/bin/env python

import os
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


def main_test(model_num):
    try:
        (train_batches,
        validation_batches, 
        pred_batches,
        TRAIN_LENGTH, 
        VAL_LENGTH, 
        PRED_LENGTH) = load_and_split_dataset_test(
            BATCH_SIZE, 
            BUFFER_SIZE, 
            load_from_tfds, 
            img_dir, 
            mask_dir, 
            train_txt_path, 
            val_txt_path, 
            input_shape[:-1],
            predData_path=os.path.join(cwd, 'test_data')
            )
        
        samples = []
        for images, masks in train_batches.take(3):
            samples.append(images[0])
            samples.append(masks[0])
        display('Dataset Sample', 'no_models', samples, 3,
                cwd, mode=0)
        loaded_model = tf.keras.models.load_model(os.path.join(cwd, 
                                                                f'model_details',
                                                                f'saved_model', 
                                                                f'model_{model_num}', 
                                                                f'saved_complete_model'), 
                                                compile=False)
                                                #, custom_objects={'mod_MeanIoU': mod_MeanIoU})
        miou = mod_MeanIoU(num_classes=output_classes, name='miou')
        opts = [optimizers.Adam(),
                optimizers.SGD(momentum=0.9, decay=5e-4, nesterov=False),
                optimizers.Adadelta()]
        loss_functions = [keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                            SparseCategoricalFocalLoss(gamma=2, from_logits=False),
                            AsymmetricLoss(from_logits=False)]
        metrics_to_compile = ['accuracy', miou]          
        loaded_model.compile(optimizer=opts[0],
                             loss=loss_functions[0],
                             metrics=metrics_to_compile)
        display_test(pred_batches, PRED_LENGTH, loaded_model, model_num)
    except:
        print("Model doesn't exist. Train and save the model first and then use this option.")

if __name__ == '__main__':
    argumentList = sys.argv[1:]   
    # Options
    options = "m:"  
    # Long options
    long_options = ["model_num="]
    
    try:
        # default values if no argument is passed.
        model_num = 0
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        # checking each argument
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-m", "--model_num"):
                model_num = int(currentValue)

        main_test(model_num)
                
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))