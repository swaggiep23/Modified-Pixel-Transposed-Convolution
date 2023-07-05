# Modified-pixel-deconvolution

This repository implements neural networks based on pixel-transposed convolution and its variations, for comparisons using Tensorflow 2. Pixel-transposed convolution is defined in [link to the paper yet to be added]
Datasets that were used for evaluation were: Pascal VOC2012 (lacked hardware resources to train for a larger number of epochs) and the Oxford-IIIT pets dataset.
Any other datasets required can be used using the config.py file
The directory structure for the project is as follows:
├───analogous_ipynb_notebook
├───dataset_1
│   ├───DatasetInfo
│   ├───JPEGImages
│   └───SegmentationClass
├───dataset_2
│   ├───DatasetInfo
│   ├───JPEGImages
│   ├───SegmentationClass
│   └───SegmentationClassFinalAug
├───model_and_training
│       model_instances.py
│       model_utilfuncs.py
│       train.py
├───model_schematics
│   ├───model_0
│   ├───model_1
│   ├───model_2
│   ├───model_3
│   └───model_4
├───saved_model
│   ├───model_0
│   │   └───saved_complete_model
│   ├────model_1
│   │   └───saved_complete_model
│   ├───model_2
│   │   └───saved_complete_model
│   ├────model_3
│   │   └───saved_complete_model
│   └───model_4
│       └───saved_complete_model
├───tensorboard
│   ├───model_0
│   ├───model_1
│   ├───model_2
│   ├───model_3
│   └───model_4
├───test_sample_results
│   ├───model_0
│   ├───model_1
│   ├───model_2
│   ├───model_3
│   └───model_4
├───training_plots
│   ├───model_0
│   ├───model_1
│   ├───model_2
│   ├───model_3
│   └───model_4
├───training_sample_results
│   ├───model_0
│   ├───model_1
│   ├───model_2
│   ├───model_3
│   └───model_4
├───utils
│       display_utils.py
│       image_proc_utils.py
│       instantiation_class.py
│       load_and_split_dataset.py
└───validation_sample_results
     ├───model_0
     ├───model_1
     ├───model_2
     ├───model_3
     └───model_4

References to be added:
Hongyang Gao's repository for pixel transposed convolution networks.
Tensorflow 2 repository, refered for subclassing the functions.
Papers discussing the 100 layer tiramisu, the deeplab xception, atrous convolution and spatial pyramidal pooling.
