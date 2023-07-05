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


def main_eval(model_num):
    try:
        loaded_model = tf.keras.models.load_model(os.path.join(cwd, 
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
        loaded_model.evaluate(validation_batches)
        show_predictions('Predictions after training', f'model_{model_num}', samples, loaded_model, 
                         cwd, dataset=validation_batches, mode=3, num=3)
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

        main_eval(model_num)
                
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))
