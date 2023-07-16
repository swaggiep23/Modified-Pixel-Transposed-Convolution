
from config_and_params import *
from model_and_training.train import *


def UNet_model(train_batches, validation_batches, input_shape, output_classes, 
               cwd, BATCH_SIZE, TRAIN_LENGTH, VAL_LENGTH, 
               VAL_SUBSPLITS, EPOCHS):

    tcl_type = 'ipixel'
    conv_option = 'sepconv2d'
    pool_option = 'conv'
    model_summary = True
    network_depth = 1
    filter_init = 2**1
    growth_rate = 2**1
    dropout = 0.2
    pyramid_layers = [1, 6, 12, 18]

    dense_layers = []
    for i in range(network_depth):
        dense_layers.append(2**(i+2))

    # Training parameters
    models = ["regularUNet", "denseUNet", "deeplab", "deeplab_pre"]
    model_select = models[0]

    model = compile_and_train_model(TRAIN_LENGTH, VAL_LENGTH,
                                    train_batches, validation_batches, 
                                    BATCH_SIZE, EPOCHS, 
                                    VAL_SUBSPLITS, output_classes,
                                    input_shape, filter_init,
                                    network_depth, tcl_type, 
                                    model_summary, dense_layers,
                                    growth_rate, dropout,
                                    conv_option, pool_option,
                                    pyramid_layers, model_select,
                                    cwd, model_name='model_0',
                                    d_format='NHWC')
    return model


def regularDenseNet(train_batches, validation_batches, input_shape, output_classes,
                          cwd, BATCH_SIZE, TRAIN_LENGTH, VAL_LENGTH,
                          VAL_SUBSPLITS, EPOCHS):

    tcl_type = 'builtin'
    conv_option = 'sepconv2d'
    pool_option = 'conv'
    model_summary = True
    network_depth = 5
    filter_init = 2**6
    growth_rate = 2**4
    dropout = 0.2
    pyramid_layers = [1, 6, 12, 18]

    dense_layers = []
    for i in range(network_depth):
        dense_layers.append(2**(i+2))

    # Training parameters
    models = ["regularUNet", "denseUNet", "deeplab", "deeplab_pre"]
    model_select = models[1]

    model = compile_and_train_model(TRAIN_LENGTH, VAL_LENGTH,
                                    train_batches, validation_batches,
                                    BATCH_SIZE, EPOCHS,
                                    VAL_SUBSPLITS, output_classes,
                                    input_shape, filter_init,
                                    network_depth, tcl_type,
                                    model_summary, dense_layers,
                                    growth_rate, dropout,
                                    conv_option, pool_option,
                                    pyramid_layers, model_select,
                                    cwd, model_name='model_0',
                                    d_format='NHWC')
    return model


def PixelDenseNet(train_batches, validation_batches, input_shape, output_classes, 
                  cwd, BATCH_SIZE, TRAIN_LENGTH, VAL_LENGTH, 
                  VAL_SUBSPLITS, EPOCHS):

    tcl_type = 'pixel'
    conv_option = 'sepconv2d'
    pool_option = 'conv'
    model_summary = True
    network_depth = 5
    filter_init = 2**6
    growth_rate = 2**4
    dropout = 0.2
    pyramid_layers = [1, 6, 12, 18]

    dense_layers = []
    for i in range(network_depth):
        dense_layers.append(2**(i+2))

    # Training parameters
    models = ["regularUNet", "denseUNet", "deeplab", "deeplab_pre"]
    model_select = models[1]

    model = compile_and_train_model(TRAIN_LENGTH, VAL_LENGTH,
                                    train_batches, validation_batches, 
                                    BATCH_SIZE, EPOCHS, 
                                    VAL_SUBSPLITS, output_classes,
                                    input_shape, filter_init,
                                    network_depth, tcl_type, 
                                    model_summary, dense_layers,
                                    growth_rate, dropout,
                                    conv_option, pool_option,
                                    pyramid_layers, model_select,
                                    cwd, model_name='model_1',
                                    d_format='NHWC')
    return model


def iPixelDenseNet(train_batches, validation_batches, input_shape, output_classes, 
                   cwd, BATCH_SIZE, TRAIN_LENGTH, VAL_LENGTH, 
                   VAL_SUBSPLITS, EPOCHS):

    tcl_type = 'ipixel'
    conv_option = 'sepconv2d'
    pool_option = 'conv'
    model_summary = True
    network_depth = 5
    filter_init = 2**6
    growth_rate = 2**4
    dropout = 0.2
    pyramid_layers = [1, 6, 12, 18]

    dense_layers = []
    for i in range(network_depth):
        dense_layers.append(2**(i+2))

    # Training parameters
    models = ["regularUNet", "denseUNet", "deeplab", "deeplab_pre"]
    model_select = models[1]

    model = compile_and_train_model(TRAIN_LENGTH, VAL_LENGTH,
                                    train_batches, validation_batches, 
                                    BATCH_SIZE, EPOCHS, 
                                    VAL_SUBSPLITS, output_classes,
                                    input_shape, filter_init,
                                    network_depth, tcl_type, 
                                    model_summary, dense_layers,
                                    growth_rate, dropout,
                                    conv_option, pool_option,
                                    pyramid_layers, model_select,
                                    cwd, model_name='model_2',
                                    d_format='NHWC')
    return model


def modifiediPixelDenseNet(train_batches, validation_batches, input_shape, output_classes, 
                           cwd, BATCH_SIZE, TRAIN_LENGTH, VAL_LENGTH, 
                           VAL_SUBSPLITS, EPOCHS):

    tcl_type = 'modified ipixel'
    conv_option = 'sepconv2d'
    pool_option = 'conv'
    model_summary = True
    network_depth = 5
    filter_init = 2**6
    growth_rate = 2**4
    dropout = 0.2
    pyramid_layers = [1, 6, 12, 18]

    dense_layers = []
    for i in range(network_depth):
        dense_layers.append(2**(i+2))

    # Training parameters
    models = ["regularUNet", "denseUNet", "deeplab", "deeplab_pre"]
    model_select = models[1]

    model = compile_and_train_model(TRAIN_LENGTH, VAL_LENGTH,
                                    train_batches, validation_batches, 
                                    BATCH_SIZE, EPOCHS, 
                                    VAL_SUBSPLITS, output_classes,
                                    input_shape, filter_init,
                                    network_depth, tcl_type, 
                                    model_summary, dense_layers,
                                    growth_rate, dropout,
                                    conv_option, pool_option,
                                    pyramid_layers, model_select,
                                    cwd, model_name='model_3', 
                                    d_format='NHWC')
    return model


def Deeplab(train_batches, validation_batches, input_shape, output_classes, 
            cwd, BATCH_SIZE, TRAIN_LENGTH, VAL_LENGTH, 
            VAL_SUBSPLITS, EPOCHS):

    tcl_type = 'modified ipixel'
    conv_option = 'sepconv2d'
    pool_option = 'conv'
    model_summary = True
    network_depth = 5
    filter_init = 2**6
    growth_rate = 2**4
    dropout = 0.2
    pyramid_layers = [1, 6, 12, 18]

    dense_layers = []
    for i in range(network_depth):
        dense_layers.append(2**(i+2))

    # Training parameters
    models = ["regularUNet", "denseUNet", "deeplab", "deeplab_pre"]
    model_select = models[2]

    model = compile_and_train_model(TRAIN_LENGTH, VAL_LENGTH,
                                    train_batches, validation_batches, 
                                    BATCH_SIZE, EPOCHS, 
                                    VAL_SUBSPLITS, output_classes,
                                    input_shape, filter_init,
                                    network_depth, tcl_type, 
                                    model_summary, dense_layers,
                                    growth_rate, dropout,
                                    conv_option, pool_option,
                                    pyramid_layers, model_select,
                                    cwd, model_name='model_4',
                                    d_format='NHWC')
    return model

