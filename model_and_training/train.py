import tensorflow as tf
import matplotlib.pyplot as plt
import functools
import datetime
import os
from tensorflow import keras
from keras import optimizers
from focal_loss import SparseCategoricalFocalLoss
from config_and_params import *
from utils.load_and_split_dataset import *
from utils.instantiation_class import *
from utils.image_proc_utils import *
from utils.display_utils import *


class mod_MeanIoU(tf.keras.metrics.MeanIoU):
    """
    Subclassing MeanIoU from tf.keras.metrics so correct for the 
    MeanIoU measurement error that comes up during model compilation.
    
    Link to the posts that discuss this: 
    https://github.com/tensorflow/tensorflow/issues/32875
    https://github.com/keras-team/keras-cv/issues/909
    https://stackoverflow.com/questions/61824470/dimensions-mismatch-error-when-using-tf-metrics-meaniou-with-sparsecategorical
    """
    def __init__(self, y_true=None, y_pred=None, num_classes=None,
                 name=None, dtype=None):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
    
    def get_config(self):
        config = super().get_config()
        config.update({
                       "num_classes": self.num_classes,                 
                       })

        return config
    

class AsymmetricLoss(tf.keras.losses.Loss):
    """
    A test loss function that didn't work.
    """
    def __init__(self, from_logits=False, gamma_neg=4.0, gamma_pos=1.0, 
                 clip=0.10, eps=1e-6):
        super().__init__()
        self.from_logits = from_logits
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.dtypes.int32)
#        Create a mask with one hot encoding for the ease of calculation.
#         y_true = tf.convert_to_tensor(tf.keras.utils.to_categorical(y_true, 
#                                                num_classes=tf.shape(y_pred)[-1]))
        y_true = tf.reshape(tf.one_hot(y_true, depth=tf.shape(y_pred)[-1]), tf.shape(y_pred))
        y_pred_sig = layers.Softmax()(y_pred) if self.from_logits else y_pred
        p = y_pred_sig
        p_m = tf.math.maximum(p-self.clip, self.eps*tf.ones(tf.shape(p)))
#             y_preds_neg = tf.clip_by_value(y_preds_neg+self.clip,
#                                            clip_value_max=1-self.eps, 
#                                            clip_value_min=self.eps)
        loss_pos = tf.math.multiply_no_nan(x=tf.math.log(tf.clip_by_value(p, 
                                                                          clip_value_min=self.eps,
                                                                          clip_value_max=1)),
                                           y=y_true)
        loss_neg = tf.math.multiply_no_nan(x=tf.math.log(tf.clip_by_value(1-p_m, 
                                                                          clip_value_min=self.eps,
                                                                          clip_value_max=1)),
                                           y=(1-y_true))
        loss = loss_pos + loss_neg
        pt = tf.math.multiply_no_nan(x=(1-p)**self.gamma_pos, y=y_true) +\
             tf.math.multiply_no_nan(x=p_m**self.gamma_neg, y=(1-y_true))
#         one_sided_gamma = tf.math.multiply_no_nan(x=self.gamma_pos, y=y_true) +\
#         tf.math.multiply_no_nan(x=self.gamma_neg, y=(1-y_true))
#         one_sided_w = pt # **one_sided_gamma
        loss=tf.math.multiply_no_nan(x=pt, y=loss)

#         return -tf.math.reduce_sum(loss, axis=None) 
        return -tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
                       "from_logits": self.from_logits,
                       "gamma_neg": self.gamma_neg,
                       "gamma_pos": self.gamma_pos,
                       "clip": self.clip,
                       "eps": self.eps     
                       })

        return config


class compile_and_train_model:
    """
    *****************
    *** Arguments ***
    *****************
    TRAIN_LENGTH: (Integer) 
        The length of the training dataset.
        
    VAL_LENGTH: (Integer) 
        The length of the validation dataset.
        
    EPOCHS: (Integer) 
        Number of epochs for TRAINING the model.
        
    VAL_SUBSPLITS: (Integer) 
        Number of subsplits for the validation dataset.
        
    output_classes: (Integer) 
        The number of output classes.
        
    input_height: (Integer) 
        Height of the input images.
        
    input_width: (Integer) 
        Width of the input images.
        
    input_channels: (Integer)
        Number of channels in the input images.
        
    filter_init: (Integer) 
        The number of filters to be used in the first layer of the U-net.
        
    network_depth: (Integer) 
        The depth of the Unet.
        
    dense_unet: (Integer) 
        To choose if we want to use Dense Unet alternative, 
        if set to False we can further choose from pixel DCNs.
        
    cl_type: (Integer)
        0 for normal convolution, 1 for ipixel convolution.
        
    tcl_type:(Integer) 
        0 for Regular deconv, 1 for Pixel deconv, 2 for iPixel deconv.
        
    model_summary:(Boolean) 
        If set to true, the model flowchart and model summary are displayed.
        
    dense_layers: (List of integers) 
        The number of dense layers in the denseblock 
        in the Unet at the depth corresponding to the list index.
                    
    growth_rate: (Integer) 
        The number of kernels/filters in each layer of the denseblock.
        
    dropout: (Integer) 
        The dropout value for the denseblock.
        
    conv_option: (String) 
        Sets the convolution type. Either 'conv2d' or 'adsconv2d'.
    
    pool_option: (String) 
        Sets the pooling option. Either 'builtin' or 'conv'.
    
    pyramid_layers: (List of integers)
        The dilation values for atrous convolution that will be used in atrousSPP.
        
    d_format: (String) 
        The data format of the input tensors - NHWC or NHCW.
    """
    def __init__(self, TRAIN_LENGTH, VAL_LENGTH, train_batches, 
                 validation_batches, BATCH_SIZE, EPOCHS, VAL_SUBSPLITS, 
                 output_classes, input_shape, filter_init, network_depth,
                 tcl_type, model_summary, dense_layers, growth_rate,
                 dropout, conv_option, pool_option, pyramid_layers,
                 model_select, cwd, model_name, d_format='NHWC'):

        # input parameters to the model, based on dataset and the computation capability.
        self.train_batches = train_batches
        self.validation_batches = validation_batches
        self.EPOCHS = EPOCHS
        self.output_classes = output_classes
        self.input_shape = input_shape
        self.filter_init = filter_init
        self.network_depth = network_depth
        self.tcl_type = tcl_type
        self.model_summary = model_summary
        self.dense_layers = dense_layers
        self.growth_rate = growth_rate
        self.dropout = dropout
        self.d_format = d_format
        self.conv_option = conv_option
        self.pool_option = pool_option
        self.pyramid_layers = pyramid_layers
        self.model_select = model_select
        self.model_name = model_name
        self.cwd = cwd

        self.STEPS_PER_EPOCH = TRAIN_LENGTH//BATCH_SIZE
        self.VALIDATION_STEPS =  VAL_LENGTH//BATCH_SIZE//VAL_SUBSPLITS #1

    def compile_model(self, cwd):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            self.miou = mod_MeanIoU(num_classes=self.output_classes, name='miou')
            opts = [optimizers.Adam(),
                    optimizers.SGD(momentum=0.9, decay=5e-4, nesterov=False),
                    optimizers.Adadelta()]
            loss_functions = [keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                              SparseCategoricalFocalLoss(gamma=2, from_logits=False),
                              AsymmetricLoss(from_logits=False)]
#                               sm.losses.jaccard_loss,
#                               sm.losses.dice_loss,
#                               sm.losses.cce_dice_loss, 
#                               sm.losses.cce_jaccard_loss,
#                               sm.losses.categorical_focal_loss,
#                               sm.losses.categorical_focal_dice_loss,
#                               sm.losses.categorical_focal_jaccard_loss]
            metrics_to_compile = ['accuracy', self.miou]
            model = modelClass(self.input_shape, self.network_depth, 
                               self.tcl_type, self.output_classes, 
                               self.filter_init, self.dense_layers, 
                               self.growth_rate, self.dropout, 
                               self.conv_option, self.pool_option, 
                               self.pyramid_layers, self.d_format)(self.model_select)            
        model.compile(optimizer=opts[0],
                      loss=loss_functions[0],
                      metrics=metrics_to_compile)
                     # run_eagerly=True) 
        if self.model_summary:
            model.summary()
            try:
                image_file_path = os.path.join(cwd, f'model_details', f'model_schematics', f'{self.model_name}',
                                               f'{self.model_name}.png')
                tf.keras.utils.plot_model(model, to_file=image_file_path, show_shapes=False)
            except:
                image_file_path = os.path.join(cwd, f'model_details', f'model_schematics', f'{self.model_name}',
                                               f'{self.model_name}.png')
                os.makedirs(os.path.dirname(image_file_path))
                tf.keras.utils.plot_model(model, to_file=image_file_path, show_shapes=False)

        self.model = model

    def train_model(self, image_list, cwd, learn_r):
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
        callback = [DisplayCallback(image_list, self.model, self.model_name, self.cwd),
                    keras.callbacks.LearningRateScheduler(functools.partial(lr_scheduler,
                                                                            lr_init=learn_r,
                                                                            max_epochs=self.EPOCHS),
                                                          verbose=1)] \
                   if inKaggle else [DisplayCallback(image_list, self.model, self.model_name, self.cwd), 
                                     keras.callbacks.TensorBoard(log_dir=os.path.join(cwd,
                                                                                      f'tensorboard', 
                                                                                      f'{self.model_name}',
                                                                                      f'{timestamp}'), 
                                     histogram_freq=1, 
                                     write_graph=True, 
                                     write_images=True,
                                     update_freq='epoch'), 
                                     keras.callbacks.LearningRateScheduler(functools.partial(lr_scheduler,
                                                                                             lr_init=learn_r,
                                                                                             max_epochs=self.EPOCHS),
                                                                           verbose=1)]
        self.model_history = self.model.fit(self.train_batches, 
                                            epochs=self.EPOCHS,
                                            steps_per_epoch=self.STEPS_PER_EPOCH,
                                            validation_steps=self.VALIDATION_STEPS,
                                            validation_data=self.validation_batches,
                                            callbacks=callback)

    def plot_train_results(self):
        train_loss = self.model_history.history['loss']
        val_loss = self.model_history.history['val_loss']
        train_iou = self.model_history.history['miou']
        val_iou = self.model_history.history['val_miou']
        plot_list = {'loss':(train_loss, 'Training Loss',
                             val_loss, 'Validation Loss'),
                     'iou':(train_iou, 'Training IOU',
                            val_iou, 'Validation IOU')}
        
        plt.figure()
        for plot_num, (key, (train_plot, train_label, val_plot, val_label))\
        in enumerate(zip(plot_list.keys(), plot_list.values())):
            plt.subplot(len(plot_list), 1, plot_num+1)
            plt.plot(self.model_history.epoch, train_plot, 
                     'r-*', label=train_label)
            plt.plot(self.model_history.epoch, val_plot,
                     'b-*', label=val_label)
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel(f"{key}".capitalize())
            # plt.ylim([0, 1])
            plt.legend()
            plt.subplots_adjust(top=0.92, bottom=0.11, left=0.125, right=0.93, hspace=0.6, wspace=0.2)
        fig_path = os.path.join(self.cwd, f'results', f'training_plots', f'{self.model_name}',
                        f'{self.model_name} - Training Metrics vs Epochs.png')
        try:
            plt.savefig(fig_path)
        except:
            os.makedirs(os.path.dirname(fig_path))
            plt.savefig(fig_path)
        # plt.show()
        
    def __call__(self, cwd, learn_r, image_list):
        self.compile_model(cwd)
        show_predictions('Pre-training Predictions', self.model_name, image_list, self.model, 
                         self.cwd, mode=1, num=3)
        self.train_model(image_list, cwd, learn_r)
        self.plot_train_results()
        show_predictions('Predictions after training', self.model_name, image_list, self.model, 
                         self.cwd, dataset=self.validation_batches, mode=2, num=3)
        # model_checkpt_path = os.path.join(cwd, f'model_details', f'saved_model', 
        #                                   f'{self.model_name}', f'saved_weights',
        #                                   f'{self.model_name}_weights.ckpt')
        complete_model_path = os.path.join(cwd, f'model_details', f'saved_model', 
                                           f'{self.model_name}', f'saved_complete_model')
        self.model.evaluate(self.validation_batches)
        try:
            # self.model.save_weights(model_checkpt_path)
            self.model.save(complete_model_path)
        except:
            # os.makedirs(os.path.dirname(model_checkpt_path))
            # self.model.save_weights(model_checkpt_path)
            os.makedirs(os.path.dirname(complete_model_path))
            self.model.save(complete_model_path)