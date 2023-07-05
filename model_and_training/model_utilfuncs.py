import tensorflow as tf
from tensorflow import keras
from keras import layers
from config_and_params import *


class BatchActivate(keras.layers.Layer):
    """
    Batchnormalization of the input followed by a ReLU activation.
    """
    def __init__(self, activation='relu'):
        super().__init__()
        self.activation =  activation
        
    def build(self, input_shape):
        if inColab:   
            self.batchnorm = layers.BatchNormalization(synchronized=True)
        else:
            self.batchnorm = layers.BatchNormalization()

        self.activate = layers.Activation(self.activation)
        super().build(input_shape)
        
    def call(self, inputs):
        return self.activate(self.batchnorm(inputs))
    
    def get_config(self):
        config = super().get_config()
        config.update({
                        "activation": self.activation
                     })
        return config

    
class adsConv2D(keras.layers.Layer):
    """
    Atrous Depthwise Separable 2D Convolution: https://arxiv.org/pdf/1802.02611.pdf
    """
    def __init__(self, filters, kernel_depthwise, padding='valid',
                 strides=1, activation=None, dilation_rate=1, 
                 kernel_initializer='he_uniform'):
        super().__init__()
        self.filters = filters
        self.kernel_depthwise = kernel_depthwise
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.kernel_initializer = kernel_initializer
        
    def build(self, input_shape):
        self.pointwise = layers.Conv2D(filters=self.filters, 
                                       kernel_size=(1, 1), 
                                       activation=self.activation, 
                                       kernel_initializer=self.kernel_initializer)
        self.depthwise = layers.DepthwiseConv2D(kernel_size=self.kernel_depthwise, 
                                                padding=self.padding,
                                                activation=self.activation,
                                                strides=self.strides,
                                                dilation_rate=self.dilation_rate, 
                                                kernel_initializer=self.kernel_initializer)
        super().build(input_shape)
        
    def call(self, inputs):
        return self.pointwise(self.depthwise(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({
                        "filters": self.filters,
                        "kernel_depthwise": self.kernel_depthwise,
                        "padding": self.padding,
                        "strides": self.strides,
                        "activation": self.activation,
                        "dilation_rate": self.dilation_rate
                     })
        return config

    
class atrousSPP(keras.layers.Layer):
    """
    Atrous Spatial Pyramidal Pooling, also implemented in Deeplabv3+:
    https://arxiv.org/pdf/1802.02611.pdf

    kernel_size: (Tuple of Integers) The kernel size used for the parallel convolutions
                  before concatenation.
    in_filters: (Integer) The number of filters in the input (due to this particular implementation).
    channel_axis: (Integer) The channel axis for all of the input tensors.
    num_outputs: (Integer) The number of channels desired in the output of this layer.
    conv_option: (String) To choose the convolution type, either "conv2d" or "adsconv2d".
    pyramid_dilations: (List of Integers) The dilation values for the parallel convolutions.
                      (for both the dimensions of a 2D image)
    """
    def __init__(self, kernel_size, in_filters, channel_axis, 
                 num_outputs, conv_option, pyramid_layers):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_filters = in_filters
        self.channel_axis = channel_axis
        self.num_outputs = num_outputs
        self.conv_option = conv_option
        self.pyramid_layers = pyramid_layers
        
    def build(self, input_shape):
        self.conv_list = []
        for dilation in self.pyramid_layers:
            self.conv_list.append(adsConv2D(self.in_filters, self.kernel_size, 
                                            padding='same', dilation_rate=dilation))
        self.convf = conv2_op(self.conv_option)(self.num_outputs, (1, 1), 
                                                padding='same')  
        super().build(input_shape)
        
    def call(self, inputs):
        return self.convf(layers.concatenate([conv(inputs) for conv in self.conv_list],
                          axis=self.channel_axis))

    def get_config(self):
        config = super().get_config()
        config.update({
                        "kernel_size": self.kernel_size,
                        "in_filters": self.in_filters,
                        "channel_axis": self.channel_axis,
                        "num_outputs": self.num_outputs,
                        "conv_option": self.conv_option,
                        "pyramid_layers": self.pyramid_layers
                     })
        return config
    

class DenseLayer(keras.layers.Layer):
    """
    This class implements the Dense layer comprised of the following: 
    Batchnormalization -> Relu activation -> Convolution (depthwise or built-in) -> Dropout.
    
    This layer is discussed in great detail in the following paper:
    https://arxiv.org/pdf/1611.09326.pdf
    """
    def __init__(self, growth_rate, kernel_dense, dropout, 
                 conv_option):
        """
        *****************
        *** Arguments ***
        *****************
        kernel_dense: (Tuple) the kernel size used for convolutions and deconvolutions.
        growth_rate: (Integer) The number of feature maps for each layer of the denseblock.
        dropout: (Integer) The dropout value for the denseblock.
        conv_option: (String) Sets the convolution type. Either 'conv2d' or 'adsconv2d'.
        """
        super().__init__()
        self.growth_rate = growth_rate
        self.kernel_dense = kernel_dense
        self.dropout = dropout
        self.conv_option = conv_option
        
    def build(self, input_shape):
        self.dropout = layers.Dropout(self.dropout)
        self.conv = conv2_op(self.conv_option)(self.growth_rate,
                                               self.kernel_dense,
                                               padding='same')
        self.batchact = BatchActivate(activation=tf.keras.layers.PReLU())
        super().build(input_shape)
        
    def call(self, inputs):
        """
        inputs: Input tensors.
        """
        return self.dropout(self.conv(self.batchact(inputs)))
    
    def get_config(self):
        config = super().get_config()
        config.update({
                    "growth_rate": self.growth_rate,
                    "kernel_dense": self.kernel_dense,
                    "dropout": self.dropout,
                    "conv_option": self.conv_option
        })
        return config
    

class dilate_tensor(keras.layers.Layer):
    def __init__(self, axis, row_shift, column_shift, 
                 upsample_factor):
        super().__init__()
        self.axis = axis
        self.row_shift = row_shift
        self.column_shift = column_shift
        self.upsample_factor = upsample_factor
        
    def call(self, inputs):
        row_shifts = [item for item in range(self.upsample_factor)]
        row_shifts.remove(self.row_shift)
        rows = tf.unstack(inputs, axis=self.axis[0])
        row_zeros = tf.zeros_like(rows[0], dtype=tf.float32)

        for step, rshift in enumerate(row_shifts):
            for index in range(len(rows), 0, -(step+1)):
                rows.insert(index-rshift, row_zeros)
        inputs = tf.stack(rows, axis=self.axis[0])

        column_shifts = [item for item in range(self.upsample_factor)]
        column_shifts.remove(self.column_shift)
        columns = tf.unstack(inputs, axis=self.axis[1])
        columns_zeros = tf.zeros_like(columns[0], dtype=tf.float32)

        for step, cshift in enumerate(column_shifts):
            for index in range(len(columns), 0, -(step+1)):
                columns.insert(index-cshift, columns_zeros)
        inputs = tf.stack(columns, axis=self.axis[1])
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
                        "axis": self.axis,
                        "row_shift": self.row_shift,
                        "column_shift": self.column_shift,
                        "upsample_factor": self.upsample_factor,
                     })
        return config
    
    
def BatchNorm(inputs):
    """
    Configures the Batchnormalization for the entire code. In case you have Tensorflow 2.12 or above installed on your machine,
    change the if else conditions to meet your requirements.
    """
    if inColab:
        outputs = layers.BatchNormalization(synchronized=True)(inputs)
    else:
        outputs = layers.BatchNormalization()(inputs)
    return outputs
  

def conv2_op(conv_option):
    if conv_option == 'conv2d':
        def conv_in(filters, kernel_size, 
                    strides=1, padding='valid',
                    dilation_rate=1, activation=None, 
                    kernel_initializer='he_uniform'):
            return tf.keras.layers.Conv2D(filters, kernel_size, 
                                          strides=strides, 
                                          padding=padding,
                                          dilation_rate=dilation_rate, 
                                          activation=activation, 
                                          kernel_initializer=kernel_initializer)
    elif conv_option == 'sepconv2d':
        def conv_in(filters, kernel_size, 
                    strides=1, padding='valid',
                    dilation_rate=1, activation=None,
                    kernel_initializer='he_uniform'):
            return tf.keras.layers.SeparableConv2D(filters, kernel_size, 
                                                   strides=strides, 
                                                   padding=padding,
                                                   dilation_rate=dilation_rate, 
                                                   activation=activation, 
                                                   kernel_initializer=kernel_initializer)
    elif conv_option == 'adsconv2d':
        def conv_in(filters, kernel_size, 
                    strides=1, padding='valid',
                    dilation_rate=1, activation=None,
                    kernel_initializer='he_uniform'):
            return adsConv2D(filters, kernel_size, 
                             strides=strides, 
                             padding=padding,
                             dilation_rate=dilation_rate,
                             activation=activation, 
                             kernel_initializer=kernel_initializer)

#     elif conv_option == 'ipixel' or 'pixel':
#         def conv_in(out_num, kernel_size, upsample_rate=2, strides=1,
#                     padding='same', dilation_rate=1, d_format='NHWC'):
#             return customCL(out_num, kernel_size, upsample_rate, strides,
#                             padding, dilation_rate, d_format)
    return conv_in


def downsample2_op(conv_option, pool_option):
    if pool_option == 'pool':
        def downsample_in(num_outputs):
            return layers.MaxPooling2D((2, 2))
    elif pool_option == 'conv':
        def downsample_in(num_outputs):
            return conv2_op(conv_option)(num_outputs, (2, 2), 
                                         padding='same', strides=2)
    return downsample_in


class customDCL:
    """
    Upsamples the input by 'upsample_rate' using the methods discussed in:
    https://arxiv.org/pdf/1705.06820.pdf
    """
    def __init__(self, dcl_type, out_num, kernel_size, 
                 upsample_rate, conv_option, d_format='NHWC'):
        """
        *****************
        *** Arguments ***
        *****************

        inputs: (4D tensor)
            Input tensor, with the dimensions.
            (batch_size, input_height, input_width, out_num)

        out_num: (integer)
            Output channel number.

        kernel_size: (2-tuple of integers)
            Convolutional kernel size.
        """
        self.dcl_type = dcl_type
        self.out_num = out_num
        self.kernel_size = kernel_size
        self.upsample_rate = upsample_rate
        self.conv_option = conv_option
        self.d_format = d_format
        self.image_dim = 2
        
    def DCL(self, inputs):
        axis =  (self.d_format.index('H'), self.d_format.index('W'))
        channel_axis = self.d_format.index('C')
        loop_inputs = inputs
        dilated_outputs = []
        for index in range(self.upsample_rate**self.image_dim):
            column_index = index%self.upsample_rate
            row_index = int(index/self.upsample_rate)%self.upsample_rate
            loop_inputs = BatchActivate(activation='relu')(loop_inputs)
            conv = conv2_op(self.conv_option)(self.out_num, 
                                         self.kernel_size, 
                                         padding='same')(loop_inputs)
            dilated_outputs.append(dilate_tensor(axis,
                                                 row_index, 
                                                 column_index,
                                                 self.upsample_rate)(conv))
            loop_inputs = conv if index==0 and self.dcl_type == 'pixel' \
                          else layers.concatenate([loop_inputs, conv],
                                                  axis=channel_axis)
        outputs = tf.add_n(dilated_outputs)
        return outputs
    
    def __call__(self, inputs):
        """
        ***************
        *** Returns ***
        ***************
        
        outputs: (4D tensor)
            Output tensor, with the dimensions.
            (batch_size, upsample_rate*input_height, upsample_rate*input_width, out_num)
        """
        return self.DCL(inputs)


class customDCL2:
    """
    Upsamples the input by 'upsample_rate', similar to `customDCL`, 
    except in this case all of the convolutions are independent of each other, unlike customDCL where 
    all of the operations are sequential.
    """
    def __init__(self, dcl_type, out_num, kernel_size, 
                 upsample_rate, conv_option, d_format='NHWC'):
        """
        *****************
        *** Arguments ***
        *****************

        inputs: (4D tensor)
            Input tensor, with the dimensions.
            (batch_size, input_height, input_width, out_num)

        out_num: (integer)
            Output channel number.

        kernel_size: (2-tuple of integers)
            Convolutional kernel size.
        """
        self.dcl_type = dcl_type
        self.out_num = out_num
        self.kernel_size = kernel_size
        self.upsample_rate = upsample_rate
        self.conv_option = conv_option
        self.d_format = d_format
        self.image_dim = 2
        
    def DCL(self, inputs):
        axis =  (self.d_format.index('H'), self.d_format.index('W'))
        channel_axis = self.d_format.index('C')
        loop_inputs = inputs
        dilated_outputs = []
        for index in range(self.upsample_rate**self.image_dim):
            column_index = index%self.upsample_rate
            row_index = int(index/self.upsample_rate)%self.upsample_rate
            loop_inputs = BatchActivate(activation='relu')(loop_inputs)
            conv = conv2_op(self.conv_option)(self.out_num, 
                                         self.kernel_size, 
                                         padding='same')(loop_inputs)
            dilated_outputs.append(dilate_tensor(axis,
                                                 row_index, 
                                                 column_index,
                                                 self.upsample_rate)(conv))
        outputs = tf.add_n(dilated_outputs)
        return outputs
    
    def __call__(self, inputs):
        """
        ***************
        *** Returns ***
        ***************
        
        outputs: (4D tensor)
            Output tensor, with the dimensions.
            (batch_size, upsample_rate*input_height, upsample_rate*input_width, out_num)
        """
        return self.DCL(inputs)