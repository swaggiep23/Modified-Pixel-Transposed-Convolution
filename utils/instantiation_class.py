import tensorflow as tf
from tensorflow import keras
from keras import layers
from config_and_params import *
from model_and_training.model_utilfuncs import *


class modelClass:
    def __init__(self, input_shape, network_depth, tcl_type, 
                 out_classnum, filtersize, dense_layers, growth_rate, 
                 dropout, conv_option, pool_option, pyramid_layers,
                 d_format='NHWC'):
        """       
        *****************
        *** Arguments ***
        *****************

        input_shape: (Tuple of integers) 
            Three dimensions representing the input shape in the form:
            (input_height, input_width, input_channels)

        network_depth: (Integer)
            Depth of the Unet.

        cl_type: (String) 
            'builtin' for normal convolution, 'ipixel' for ipixel convolution.

        tcl_type: (String) 
            'builtin' for normal deconvolution, 'pixel' for pixel deconvolution,
            'ipixel' ipixel deconvolution.

        out_classnum: (Integer) 
            The number of output classes.

        filtersize: (Integer) 
            The number of filters/channels for the first layer of the Unet.

       dense_layers: (List of integers) 
           The number of dense layers in the denseblock in the Unet
           at the depth corresponding to the list index.

        growth_rate: (Integer) 
            The number of kernels/filters in each layer of
            the denseblock.

        dropout: (Integer) 
            The dropout value for the denseblock.

        conv_option: (String) 
            Sets the convolution type. Either 'conv2d' or 'adsconv2d'.

        pool_option: (String) 
            Sets the pooling option. Either 'builtin' or 'conv'.

        pyramid_layers: (List of integers) 
            The dilation values for atrous convolution that will be
            used in atrousSPP.

        d_format: (String) 
            The data format of the input tensors - NHWC or NHCW.
        """
        self.input_shape = input_shape
        self.network_depth = network_depth
        self.tcl_type = tcl_type
        self.out_classnum = out_classnum
        self.filtersize = filtersize
        self.dense_layers = dense_layers
        self.growth_rate = growth_rate
        self.dropout = dropout
        self.conv_option = conv_option
        self.pool_option = pool_option
        self.pyramid_layers = pyramid_layers
        self.d_format = d_format
    
    def DenseBlock(self, inputs, dense_layers, growth_rate,
                   kernel_dense, dropout, channel_axis, conv_option):
        layer_input = inputs
        dense = []
        for dense_layer in range(dense_layers):
            denseouts = DenseLayer(growth_rate, kernel_dense, 
                                   dropout, conv_option)(layer_input)
            dense.append(denseouts)
            layer_input = layers.concatenate([layer_input, denseouts],
                                             axis=channel_axis)
        dense_final = layers.concatenate([dense_item for dense_item in dense],
                                         axis=channel_axis)
        dense_concat = layers.concatenate([inputs, dense_final],
                                          axis=channel_axis)
        return dense_final, dense_concat
    
    def ResnetBlock(self, inputs, repeats, filters,
                    kernel_dense, dropout, channel_axis, conv_option):
        layer_input = inputs
        for dense_layer in range(repeats):
            denseout1 = DenseLayer(filters, kernel_dense, 
                                   dropout, conv_option)(layer_input)
            denseout2 = DenseLayer(filters, kernel_dense, 
                                   dropout, conv_option)(denseout1)   
            layer_input = layers.Add()([layer_input, denseout2])
        return layer_input

    def TransitionDown(self, inputs, num_outputs, dropout,
                       conv_option, pool_option):
        conv = DenseLayer(num_outputs, (1, 1), dropout, conv_option)(inputs)
        conv = downsample2_op(conv_option, pool_option)(num_outputs)(conv)
        return conv

    def TransitionUp(self, inputs, tcl_type, num_outputs, conv_option,  
                     kernel_size_u, upsample_rate):
        if tcl_type == 'builtin':
            convt = layers.Conv2DTranspose(num_outputs, upsample_rate, 
                                           strides=upsample_rate, padding='same',
                                           kernel_initializer='he_uniform')(inputs)
#             convt = layers.UpSampling2D(size=upsample_rate, data_format="channels_last", 
#                                         interpolation="bilinear")(inputs)
        elif tcl_type == 'ipixel' or tcl_type == 'pixel':
            convt = customTCL(tcl_type, num_outputs, kernel_size_u, upsample_rate,
                              conv_option, d_format='NHWC')(inputs)
        elif tcl_type == 'modified ipixel':
            convt = customTCL2(tcl_type, num_outputs, kernel_size_u, upsample_rate, 
                               conv_option, d_format='NHWC')(inputs)
        return convt

    def down_block_pixel(self, inputs, down_outputs, kernel_size_d,
                         kernel_size_d2, channel_axis, dense_layers, layer_index,
                         growth_rate, dropout, conv_option, pool_option, 
                         pyramid_layers, isFinal=False):
        """
        *****************
        *** Arguments ***
        *****************

        down_outputs: (List of tensors) 
            The list of tensors that contains all the downsampled inputs, which are to be 
            concatenated to the processed input tensor in the up block.

        kernel_size_u: (Tuple)
            The kernel size used for convolutions and deconvolutions.

        kernel_size_d2: (Tuple) 
            The kernel size used for convolutions and deconvolutions for ipixelcl.
            
        ***************
        *** Returns ***
        ***************
        
        dense_features: (4-D Tensor)
            The processed tensor from the respective UNet layer,
            to be sent to the next  upsample layer.
        """
        if not isFinal:
            dense_features, dense_concat = self.DenseBlock(inputs, 
                                                           dense_layers[layer_index], 
                                                           growth_rate, 
                                                           kernel_size_d, 
                                                           dropout, 
                                                           channel_axis, 
                                                           conv_option)
            down_outputs.append(dense_concat)
            layer_out = self.TransitionDown(dense_concat, 
                                            dense_concat.shape[channel_axis],
                                            dropout, 
                                            conv_option,
                                            pool_option)
        else:
            conv = conv2_op(conv_option)(inputs.shape[channel_axis], 
                                         (1, 1), 
                                         padding='same')(inputs)
            conv = BatchActivate()(conv)
            aspp_out = atrousSPP(kernel_size_d, 
                                 inputs.shape[channel_axis], 
                                 channel_axis, 
                                 inputs.shape[channel_axis], 
                                 conv_option, 
                                 pyramid_layers)(inputs)
            layer_out = layers.concatenate([conv, aspp_out], axis=channel_axis)
            layer_out = conv2_op(conv_option)(inputs.shape[channel_axis], 
                                              kernel_size_d, 
                                              padding='same')(layer_out)
            layer_out = BatchActivate()(layer_out)
        return layer_out

    def up_block_pixel(self, inputs, down_outputs, kernel_size_u, 
                       kernel_size_u2, channel_axis, out_classnum, tcl_type, 
                       dense_layers, layer_index, growth_rate, dropout,
                       conv_option, isFinal=False):
        """
        *****************
        *** Arguments ***
        *****************

        down_outputs: (List of tensors) 
            The list of tensors that contains all the downsampled inputs, which are to be 
            concatenated to the processed input tensor in the up block.

        kernel_size_u: (Tuple)
            The kernel size used for convolutions and deconvolutions.

        kernel_size_u2: (Tuple) 
            The kernel size used for convolution in the final layer.

        isFinal: (Boolean) 
            To check if the up_block is the final block of the Unet encoder
            and thereby uses out_classnum in its final convolution.

        ***************
        *** Returns ***
        ***************
        
        dense_features: (4-D Tensor)
            The processed tensor from the respective UNet layer,
            to be sent to the next downsample or upsample layer.
        """
        input_channels = inputs.shape[channel_axis]
        upsample_rate = 2
        convt = self.TransitionUp(inputs, 
                                  tcl_type,
                                  input_channels, 
                                  conv_option,
                                  kernel_size_u, 
                                  upsample_rate)
        convt = layers.concatenate([convt, down_outputs[layer_index]], axis=channel_axis)
#         convt = self.TransitionUp(inputs, 
#                                   tcl_type, 
#                                   down_outputs[layer_index].shape[channel_axis],
#                                   conv_option,
#                                   kernel_size_u, 
#                                   upsample_rate)
#         convt = layers.Add()([convt, down_outputs[layer_index]])
        layer_out = DenseLayer(dense_layers[layer_index]*growth_rate,
                               kernel_size_u, 
                               dropout, 
                               conv_option)(convt)
        return layer_out
    
    def down_block_regular(self, inputs, down_outputs, kernel_size_d, 
                           kernel_size_d2, channel_axis, dense_layers, layer_index, 
                           growth_rate, dropout, conv_option, pool_option, 
                           pyramid_layers, isFinal=False):
        """
        *****************
        *** Arguments ***
        *****************

        down_outputs: (List of tensors) 
            The list of tensors that contains all the downsampled inputs, which are to be 
            concatenated to the processed input tensor in the up block.

        kernel_size_u: (Tuple)
            The kernel size used for convolutions and deconvolutions.

        kernel_size_d2: (Tuple) 
            The kernel size used for convolutions and deconvolutions for ipixelcl.
            
        ***************
        *** Returns ***
        ***************
        
        dense_features: (4-D Tensor)
            The processed tensor from the respective UNet layer,
            to be sent to the next  upsample layer.
        """
        if not isFinal:
            conv = layers.Conv2D(self.filtersize*2**(layer_index), 
                                kernel_size_d, 
                                padding='same',
                                kernel_initializer='he_uniform')(inputs)
            conv = layers.Conv2D(self.filtersize*2**(layer_index), 
                                kernel_size_d, 
                                padding='same',
                                kernel_initializer='he_uniform')(conv)
            down_outputs.append(conv)
            layer_out = self.TransitionDown(conv, 
                                            conv.shape[channel_axis],
                                            dropout, 
                                            conv_option,
                                            pool_option)
        else:
            conv = layers.Conv2D(self.filtersize*2**(layer_index), 
                                kernel_size_d, 
                                padding='same',
                                kernel_initializer='he_uniform')(inputs)
            layer_out = layers.Conv2D(self.filtersize*2**(layer_index), 
                                kernel_size_d, 
                                padding='same',
                                kernel_initializer='he_uniform')(conv)
        return layer_out

    def up_block_regular(self, inputs, down_outputs, kernel_size_u, 
                         kernel_size_u2, channel_axis, out_classnum, tcl_type, 
                         dense_layers, layer_index, growth_rate, dropout,
                         conv_option, isFinal=False):
        """
        *****************
        *** Arguments ***
        *****************

        down_outputs: (List of tensors) 
            The list of tensors that contains all the downsampled inputs, which are to be 
            concatenated to the processed input tensor in the up block.

        kernel_size_u: (Tuple)
            The kernel size used for convolutions and deconvolutions.

        kernel_size_u2: (Tuple) 
            The kernel size used for convolution in the final layer.

        isFinal: (Boolean) 
            To check if the up_block is the final block of the Unet encoder
            and thereby uses out_classnum in its final convolution.

        ***************
        *** Returns ***
        ***************
        
        dense_features: (4-D Tensor)
            The processed tensor from the respective UNet layer,
            to be sent to the next downsample or upsample layer.
        """
        input_channels = inputs.shape[channel_axis]
        upsample_rate = 2
        convt = self.TransitionUp(inputs, 
                                  tcl_type, 
                                  input_channels, 
                                  conv_option,
                                  kernel_size_u, 
                                  upsample_rate)
        convt = layers.concatenate([convt, down_outputs[layer_index]], axis=channel_axis)
#         convt = self.TransitionUp(inputs, 
#                                   tcl_type, 
#                                   down_outputs[layer_index].shape[channel_axis],
#                                   conv_option,
#                                   kernel_size_u, 
#                                   upsample_rate)
#         convt = layers.Add()([convt, down_outputs[layer_index]])
        layer_out = layers.Conv2D(self.filtersize*2**(layer_index), 
                                  kernel_size_u, 
                                  padding='same',
                                  kernel_initializer='he_uniform')(convt)
        return layer_out

    def UNet(self, select):
        """
        ***************
        *** Returns ***
        ***************

        pixeldenseunet: (tf.keras.Model object)
            The model object for further compilation and training.
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        channel_axis = self.d_format.index('C') 
        down_outputs = []
        kernel_size_d = (3, 3)
        kernel_size_d2 = (2, 2)
        kernel_size_u = (3, 3)
        kernel_size_u2 = (1, 1)

        outputs = layers.Conv2D(self.filtersize, 
                                (1, 1), 
                                padding='same',
                                kernel_initializer='he_uniform')(inputs)
        outputs = BatchNorm(outputs)
        for layer_index in range(self.network_depth):  
            isFinal = False if layer_index != self.network_depth-1 else True
            if select=="regular_UNet":
                outputs = self.down_block_regular(outputs, down_outputs, 
                                                  kernel_size_d, kernel_size_d2,
                                                  channel_axis, self.dense_layers, 
                                                  layer_index, self.growth_rate, 
                                                  self.dropout, self.conv_option,
                                                  self.pool_option, self.pyramid_layers,
                                                  isFinal=isFinal)
            elif select=="pixeldense":
                outputs = self.down_block_pixel(outputs, down_outputs, 
                                                kernel_size_d, kernel_size_d2,
                                                channel_axis, self.dense_layers, 
                                                layer_index, self.growth_rate, 
                                                self.dropout, self.conv_option,
                                                self.pool_option, self.pyramid_layers,
                                                isFinal=isFinal)
        for layer_index in range(self.network_depth-2, -1, -1):
            isFinal = False if layer_index !=0 else True
            if select=="regular_UNet":
                outputs = self.up_block_regular(outputs, down_outputs, 
                                                kernel_size_u, kernel_size_u2,
                                                channel_axis, self.out_classnum,
                                                self.tcl_type, self.dense_layers,
                                                layer_index, self.growth_rate,
                                                self.dropout, self.conv_option,
                                                isFinal=isFinal)
            elif select=="pixeldense":
                outputs = self.up_block_pixel(outputs, down_outputs, 
                                              kernel_size_u, kernel_size_u2,
                                              channel_axis, self.out_classnum,
                                              self.tcl_type, self.dense_layers,
                                              layer_index, self.growth_rate,
                                              self.dropout, self.conv_option,
                                              isFinal=isFinal)
        outputs = layers.Conv2D(self.out_classnum,
                                kernel_size_u2, 
                                padding='same', 
                                kernel_initializer='he_uniform')(outputs)
        outputs = layers.Softmax()(outputs)
        if select=="regular_UNet":
            name = 'regular_UNet'
        elif select=="pixeldense":
            name = "pixeldenseUNet"
        unet = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
        return unet
    
    def Deeplab_repeat_block(self, inputs, repeats, filters,
                             kernel_size, add_with_conv):
        layer_input = inputs
        final_stride = 2 if add_with_conv else 1
        for layer in range(repeats):
            conv = layers.SeparableConv2D(filters, 
                                          kernel_size, 
                                          padding='same')(layer_input)
            conv = BatchActivate()(conv)
            conv = layers.SeparableConv2D(filters, 
                                          kernel_size, 
                                          padding='same')(conv)
            conv = BatchActivate()(conv)
            conv = layers.SeparableConv2D(filters, 
                                          kernel_size, 
                                          strides=final_stride,
                                          padding='same')(conv)            
            if add_with_conv:
                conv_res = layers.Conv2D(filters, 
                                         (1, 1), 
                                         strides=final_stride,
                                         padding='same')(layer_input)
            else: 
                conv_res = layer_input
            layer_input = layers.Add()([conv, conv_res])
            layer_input = BatchActivate()(layer_input)
        return layer_input
    
    def Deeplab(self, pretrained):
        """
        Link to the research paper that describes the structure of deeplab xception - 
        https://arxiv.org/pdf/1610.02357.pdf
        
        Source code - 
        https://github.com/tensorflow/models/blob/master/research/deeplab/core/xception.py
        *****************
        *** Arguments ***
        *****************
        pretrained: (boolean) Selects if the entire network is to be trained or just the decoder part.
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        channel_axis = self.d_format.index('C') 
        kernel_size_d = (3, 3)
        kernel_size_d2 = (2, 2)
        kernel_size_u = (3, 3)
        kernel_size_u2 = (1, 1)
        
        if not pretrained:
            # Entry flow
            outputs = layers.Conv2D(32, kernel_size=kernel_size_d, padding='same',
                                    kernel_initializer='he_uniform')(inputs)
            outputs = BatchNorm(outputs)
            outputs = layers.Conv2D(64, kernel_size=kernel_size_d, padding='same', 
                                    kernel_initializer='he_uniform')(outputs)
            outputs = BatchNorm(outputs)
            outputs = self.Deeplab_repeat_block(outputs, repeats=1, filters=128,
                                                kernel_size=kernel_size_d, add_with_conv=True)
            outputs = self.Deeplab_repeat_block(outputs, repeats=1, filters=256,
                                                kernel_size=kernel_size_d, add_with_conv=True)
            downsampled_out = outputs
            outputs = self.Deeplab_repeat_block(outputs, repeats=1, filters=728,
                                                kernel_size=kernel_size_d, add_with_conv=True)
            # Middle flow
            outputs = self.Deeplab_repeat_block(outputs, repeats=16, filters=728,
                                                kernel_size=kernel_size_d, add_with_conv=False)
            # Exit flow
            outputs = self.Deeplab_repeat_block(outputs, repeats=1, filters=1024,
                                                kernel_size=kernel_size_d, add_with_conv=True)
            outputs =layers.SeparableConv2D(1536, kernel_size=kernel_size_d, padding='same')(outputs)
            outputs = BatchActivate()(outputs)
            outputs = layers.SeparableConv2D(1536, kernel_size=kernel_size_d, padding='same')(outputs)
            outputs = BatchActivate()(outputs)
            outputs = layers.SeparableConv2D(2048, kernel_size=kernel_size_d, padding='same')(outputs) 
            outputs = BatchActivate()(outputs)
        else:
            base_model = keras.applications.Xception(
                                                     include_top=False,
                                                     weights="imagenet",
                                                     input_shape=self.input_shape,
                                                     input_tensor=inputs
                                                    )
            base_model.trainable=False
#             inputs = keras.Input(shape=self.input_shape)
#             x = base_model(inputs, training=False)
            outputs = base_model.get_layer('block14_sepconv2_act').output
            downsampled_out = base_model.get_layer('block4_sepconv2_bn').output
        
        # ASPP module
        aspp_out = atrousSPP(kernel_size_d, 256, channel_axis, 256, 
                             self.conv_option, self.pyramid_layers)(outputs)
#         aspp_out = layers.UpSampling2D(size=(4, 4), data_format="channels_last", 
#                                        interpolation="bilinear")(aspp_out)
        aspp_out = self.TransitionUp(aspp_out, 'builtin', 256, (4, 4), self.conv_option,
                                     upsample_rate=4)
        
        # Decoder
        conv = layers.Conv2D(48, (1, 1), padding='same')(downsampled_out)
        conv = BatchNorm(conv)
        outputs = layers.concatenate([conv, aspp_out], axis=channel_axis)
        outputs = layers.Conv2D(256, kernel_size_u, padding='same')(outputs)
        outputs = BatchNorm(outputs)
        if not pretrained:
#             outputs = layers.UpSampling2D(size=(4, 4), data_format="channels_last",
#                                           interpolation="bilinear")(outputs)
            outputs = self.TransitionUp(outputs, 'builtin', 256, (4, 4), self.conv_option,
                                        upsample_rate=4)
        else:
#             outputs = layers.UpSampling2D(size=(8, 8), data_format="channels_last",
#                                           interpolation="bilinear")(outputs)
            outputs = self.TransitionUp(outputs, 'builtin', 256, (8, 8), self.conv_option,
                                        upsample_rate=8)
        outputs = layers.Conv2D(self.out_classnum, (1, 1), padding='same', 
                                kernel_initializer='he_uniform')(outputs)
        outputs = layers.Softmax()(outputs)
        if not pretrained:
            name = 'deeplab'
        else:
            name = 'deeplab_pretrained'
        deeplab = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
        return deeplab
    
    def __call__(self, select_model):
        if select_model=="pixeldense":
            return self.UNet(select="pixeldense")
        elif select_model=="regular_UNet":
            return self.UNet(select="regular_UNet")
        elif select_model=="deeplab":
            return self.Deeplab(pretrained=False)
        elif select_model=="deeplab_pre":
            return self.Deeplab(pretrained=True)