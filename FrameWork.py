from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tensorflow.keras.utils import to_categorical as np_utils
import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras import backend as K
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os
import tensorflow as tf
from matplotlib.colors import ListedColormap, to_rgba
import cv2
from skimage import measure
from matplotlib.patches import Circle
from matplotlib import cm
import netCDF4 as nc
import xarray as xr
from scipy.ndimage import gaussian_filter

def normalize_tuple(value, n, name):
    """Transforms a single int or iterable of ints into an int tuple.
    
    Args:
        value: The value to validate and convert. Could an int, or any iterable
            of ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. "strides" or
            "kernel_size". This is only used to format error messages.
    
    Returns:
        A tuple of n integers.
    
    Raises:
        ValueError: If something else than an int/long or iterable thereof was
            passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(f'The `{name}` argument must be a tuple of {n} integers. '
                           f'Received: {value}')
        if len(value_tuple) != n:
            raise ValueError(f'The `{name}` argument must be a tuple of {n} integers. '
                           f'Received: {value}')
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(f'The `{name}` argument must be a tuple of {n} integers. '
                               f'Received: {value}')
        return value_tuple

def normalize_data_format(value):
    """Normalizes the data format value.
    
    Args:
        value: The data format value to normalize.
    
    Returns:
        A string, either 'channels_first' or 'channels_last'.
    
    Raises:
        ValueError: If the value is not 'channels_first' or 'channels_last'.
    """
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                       '"channels_first", "channels_last". Received: ' +
                       str(value))
    return data_format

# Define data directory
DATA_DIR = r'C:/Users/jalla/Downloads/Eddy/Sea'

# Load and preprocess data
SSH_train = np.load(os.path.join(DATA_DIR, 'filtered_SSH_train_data.npy'))[:,:80,:84]

# Load segmentation masks
Seg_train = np.load(os.path.join(DATA_DIR, 'train_groundtruth_Segmentation.npy'))[:,:80,:84]

# Inspect segmentation data structure
print("\nSegmentation data inspection:")
print("Seg_train shape:", Seg_train.shape)
print("Seg_train dtype:", Seg_train.dtype)
print("Seg_train min/max:", np.min(Seg_train), np.max(Seg_train))
print("Unique values in Seg_train:", np.unique(Seg_train))

# Check if we need to load separate masks
if len(np.unique(Seg_train)) <= 2:  # If only binary values exist
    print("\nBinary masks detected. Loading separate masks...")
    # Load individual masks if they exist
    try:
        anti_mask = np.load(os.path.join(DATA_DIR, 'anticyclonic_mask.npy'))
        cyc_mask = np.load(os.path.join(DATA_DIR, 'cyclonic_mask.npy'))
        
        # Create proper label map
        Seg_train = np.zeros_like(anti_mask, dtype=np.uint8)
        Seg_train[anti_mask == 1] = 2   # Label for anticyclonic
        Seg_train[cyc_mask == 1] = 1    # Label for cyclonic
        # Background is already 0
        
        print("Created new label map with values:", np.unique(Seg_train))
    except FileNotFoundError:
        print("Warning: Separate masks not found. Using original segmentation.")

# Ensure correct shape
Seg_train = Seg_train.reshape(-1, 80, 84)  # Shape: (samples, 80, 84)

# Convert to categorical
Seg_train_categor = np_utils(Seg_train, num_classes=3)  # Shape: (samples, 80, 84, 3)

# Verify class distribution
print("\nClass distribution (sum across all samples):")
print("Class 0 (background):", np.sum(Seg_train_categor[:,:,:,0]))
print("Class 1 (cyclonic):", np.sum(Seg_train_categor[:,:,:,1]))
print("Class 2 (anticyclonic):", np.sum(Seg_train_categor[:,:,:,2]))

# Visual check of labels
idx = 0  # first sample
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(Seg_train_categor[idx,:,:,0])
plt.title("Background")
plt.colorbar()
plt.subplot(132)
plt.imshow(Seg_train_categor[idx,:,:,1])
plt.title("Cyclonic")
plt.colorbar()
plt.subplot(133)
plt.imshow(Seg_train_categor[idx,:,:,2])
plt.title("Anticyclonic")
plt.colorbar()
plt.savefig('./eddydlv3net/label_check.png')
plt.close()

# Add channel dimension to SSH data
SSH_train = SSH_train[..., np.newaxis]  # Shape: (samples, 80, 84, 1)

def weighted_categorical_crossentropy(weights):
    weights = tf.keras.backend.variable(weights)
    
    def loss(y_true, y_pred):
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        return -tf.keras.backend.sum(loss, -1)
    
    return loss

# Define class weights (background=0.2, eddy classes=1.0)
class_weights = np.array([0.2, 1.0, 1.0])
loss_fn = weighted_categorical_crossentropy(class_weights)

# Debug prints for data shapes
print("SSH_train.shape:", SSH_train.shape)
print("Seg_train.shape:", Seg_train.shape)

# Update reshaping for new dimensions
num_train_samples = Seg_train.shape[0]

# Debug prints for categorical data
print("Seg_train_categor.shape:", Seg_train_categor.shape)
print("Unique values in first sample:", np.unique(Seg_train_categor[0]))

# Update visualization code to use new dimensions
randindex = np.random.randint(0,len(SSH_train))
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.imshow(SSH_train[randindex], cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
plt.axis('off')
plt.title('SSH', fontsize=24)

plt.subplot(122)
plt.imshow(Seg_train[randindex], cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
plt.axis('off')
plt.title('groundtruth Segmentation', fontsize=24)
import os
os.makedirs('./eddydlv3net', exist_ok=True)  # make sure directory exists
plt.savefig('./eddydlv3net/heatmap.png')     # save the heatmap

# Update model parameters for new dimensions
height = 80
width = 84
nf = 128
nbClass = 3
ker = 3

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = normalize_tuple(output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return tf.image.resize(inputs, 
                                 (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                 method='bilinear')
        else:
            return tf.image.resize(inputs, 
                                 (self.output_size[0],
                                                       self.output_size[1]),
                                 method='bilinear')

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):

    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):

    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand', dilation_rate=3)(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project', dilation_rate=3)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def Deeplabv3(input_tensor=None, input_shape=(80, 84, 1), classes=3, backbone='xception', OS=16, alpha=1.):

    if K.backend() != 'tensorflow':
        raise RuntimeError('The Deeplabv3+ model is only available with '
                           'the TensorFlow backend.')

    if not (backbone in {'xception', 'mobilenetv2'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`  or `mobilenetv2` ')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        x = Conv2D(32, (3, 3), strides=(2, 2),
                   name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = Activation('relu')(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = Activation('relu')(x)

        x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)

        # convert xception last Sep Conv layer in each block to maxpooling
        # x = _xception_block(x, [128, 128], 'enentry_flow_block1',
        #                     skip_connection_type='conv', stride=2,
        #                     depth_activation=False)
        # x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

        x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                   skip_connection_type='conv', stride=2,
                                   depth_activation=False, return_skip=True)

        # convert xception last Sep Conv layer in each block to maxpooling
        # x, skip1 = _xception_block(x, [256, 256], 'entry_flow_block2',
        #                            skip_connection_type='conv', stride=2,
        #                            depth_activation=False, return_skip=True)
        # x, skip1 = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

        x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                            skip_connection_type='conv', stride=entry_block3_stride,
                            depth_activation=False)

        # convert xception last Sep Conv layer in each block to maxpooling
        # x = _xception_block(x, [728, 728], 'entry_flow_block3',
        #                     skip_connection_type='conv', stride=entry_block3_stride,
        #                     depth_activation=False)
        # x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        for i in range(16):
            x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                depth_activation=False)

        x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                            skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                            depth_activation=False)

        # convert xception last Sep Conv layer in each block to maxpooling
        # x = _xception_block(x, [728, 1024], 'exit_flow_block1',
        #                     skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
        #                     depth_activation=False)
        # x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

        x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                            skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                            depth_activation=True)

        # convert xception second Sep Conv layer in each block to maxpooling
        # x = _xception_block(x, [1536, 2028], 'exit_flow_block2',
        #                     skip_connection_type='none', stride=1, rate=exit_block_rates[1],
        #                     depth_activation=True)
        # x = GlobalAveragePooling2D()(x)
    else:
        OS = 8
        first_block_filters = _make_divisible(128 * alpha, 8)
        x = Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2), padding='same',
                   use_bias=False, name='Conv')(img_input)
        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = Activation(relu6, name='Conv_Relu6')(x)

        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=1, skip_connection=False)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                                expansion=6, block_id=3, skip_connection=False)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=6, block_id=4, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,  # 1!
                                expansion=6, block_id=6, skip_connection=False)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=7, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=8, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=9, skip_connection=True)

        x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=10, skip_connection=False)
        x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=11, skip_connection=True)
        x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=12, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                expansion=6, block_id=13, skip_connection=False)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=14, skip_connection=True)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=15, skip_connection=True)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=16, skip_connection=False)


    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                            int(np.ceil(input_shape[1] / 4))))(x)
        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation('relu')(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)


    if classes == 3:
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)
    
    # Ensure correct output shape and activation
    x = Reshape((height, width, nbClass))(x)
    x = Activation('softmax')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='deeplabv3_plus')

    return model


eddydlv3net = Deeplabv3(input_shape=(height, width, 1))
# eddydlv3net = multi_gpu_model(eddydlv3net, gpus=2)  # Commented out multi-GPU usage
eddydlv3net.summary()


unipue, counts = np.unique(Seg_train, return_counts=True)
dict(zip(unipue, counts))

freq = [np.sum(counts)/j for j in counts]
weightsSeg = [f/np.sum(freq) for f in freq]

###loss function

smooth = 1e-6

def dice_coef_class(y_true, y_pred, class_index):
    # Extract class channels from true labels
    y_true_class = y_true[..., class_index]  # (batch, 80, 84)
    
    # Reshape prediction to match true labels shape
    y_pred = tf.reshape(y_pred, tf.shape(y_true))  # Match y_true shape
    y_pred_class = y_pred[..., class_index]        # (batch, 80, 84)
    
    # Debug shapes
    tf.print("dice_class: y_true", tf.shape(y_true_class), "y_pred", tf.shape(y_pred_class))
    
    # Flatten without squeezing
    y_true_f = K.flatten(y_true_class)
    y_pred_f = K.flatten(y_pred_class)
    
    # Calculate intersection and dice coefficient
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_anti(y_true, y_pred):
    return dice_coef_class(y_true, y_pred, 2)  # Anticyclonic

def dice_coef_cyc(y_true, y_pred):
    return dice_coef_class(y_true, y_pred, 1)  # Cyclonic

def dice_coef_nn(y_true, y_pred):
    return dice_coef_class(y_true, y_pred, 0)  # Non-eddy

def mean_dice_coef(y_true, y_pred):
    return (dice_coef_anti(y_true, y_pred) + dice_coef_cyc(y_true, y_pred) + dice_coef_nn(y_true, y_pred)) / 3

def weighted_mean_dice_coef(y_true, y_pred):
    return (0.35 * dice_coef_anti(y_true, y_pred) +
            0.62 * dice_coef_cyc(y_true, y_pred) +
            0.03 * dice_coef_nn(y_true, y_pred))

def dice_coef_loss(y_true, y_pred):
    # Ensure 4D shapes at the loss level
    y_true = tf.ensure_shape(y_true, [None, None, None, 3])
    y_pred = tf.ensure_shape(y_pred, [None, None, None, 3])
    return 1.0 - weighted_mean_dice_coef(y_true, y_pred)

def precision(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positive / (predicted_positive + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positive / (possible_positive + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    y_pred = K.round(K.clip(y_pred, 0, 1))
    y_true = K.round(K.clip(y_true, 0, 1))

    def zero_score():
        return tf.constant(0.0, dtype=tf.float32)

    def compute_score():
        tp = K.sum(y_true * y_pred)
        precision = tp / (K.sum(y_pred) + K.epsilon())
        recall = tp / (K.sum(y_true) + K.epsilon())
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + K.epsilon())

    return tf.cond(K.sum(y_true) == 0, zero_score, compute_score)

#def fmeasure(y_true, y_pred):
#    return fbeta_score(y_true, y_pred, beta=1)

# Compile model with weighted loss
eddydlv3net.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=['categorical_accuracy', mean_dice_coef,
             dice_coef_anti, dice_coef_cyc, dice_coef_nn,
             weighted_mean_dice_coef, precision, recall, fbeta_score])

#earl = EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=80, verbose=1, mode='auto')
modelcheck = ModelCheckpoint('./eddydlv3net/eddynet.weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
reducecall = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1, mode='auto', min_delta=1e-30, min_lr=1e-30)

# Train model without validation split
hiseddydlv3net = eddydlv3net.fit(
    SSH_train,
    Seg_train_categor,
    batch_size=4,
    epochs=300,
    verbose=1,
    callbacks=[modelcheck, reducecall],
    validation_split=0.0,  # Temporarily disable validation
    shuffle=True
)

# Save the trained model (with architecture + weights)
print("\nSaving model weights...")
eddydlv3net.save_weights('./eddydlv3net/eddynet.weights.h5')
print("✅ Model saved successfully to: ./eddydlv3net/eddynet.weights.h5")

plt.figure(figsize=(10, 10))
plt.semilogy(eddydlv3net.history.history['loss'])
# plt.semilogy(eddydlv3net.history.history['val_loss'])
# plt.semilogy(eddydlv3net.history.history['val_categorical_accuracy'])
plt.title('EddyDLv3 Training Loss', fontsize=20)
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train_loss', 'val_loss', 'val_acc'], loc='center right')
plt.legend(['train_loss'], loc='center right')
plt.savefig('./eddydlv3net/EddyDLv3Loss.png')
#plt.show()

#performance on train dataset

randindex = np.random.randint(0,len(SSH_train))
predictedSEGM = eddydlv3net.predict(np.reshape(SSH_train[randindex,:,:], (1, height, width)))
predictedSEGMimage = predictedSEGM[0].argmax(axis=-1)  # shape: (80, 84)

plt.figure(figsize=(20, 10))
plt.subplot(131)
plt.imshow(SSH_train[randindex], cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
plt.axis('off')
plt.title('SSH', fontsize=24)

plt.subplot(132)
plt.imshow(predictedSEGMimage, cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
plt.axis('off')
plt.title('EddyDLv3 Method', fontsize=24)

plt.subplot(133)
plt.imshow(Seg_train[randindex], cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
plt.axis('off')
plt.title('PET Method', fontsize=24)
plt.savefig('./eddydlv3net/traindata.png')
#plt.show()

#metrics on test dataset

preds = eddydlv3net.evaluate(SSH_train, Seg_train_categor)
print('loss: %s,'%preds[0], 'accuracy: %s,'%preds[1], 'mean_dice_coef: %s,'%preds[2], 'weighted_mean_dice_coef: %s,'%preds[3],
      'MD cyc: %s'%preds[4], 'MD NE: %s'%preds[5], 'MD anti: %s'%preds[6],
      'precision: %s,'%preds[7], 'recall: %s,'%preds[8], 'f1_score: %s'%preds[9])

# Discrete colormap for segmentation classes: [background, cyclonic, anti-cyclonic]
cmap = ListedColormap(["purple", "blue", "red"])  # 0: bg, 1: cyclonic, 2: anti-cyclonic

# Predict all
def save_discrete_colormap_predictions(predictions, nc_path, out_dir):
    """
    Saves heatmap+overlayed-prediction images to out_dir/predicted_eddies.
    Uses xarray for robust NetCDF handling and proper land masking.
    """
    os.makedirs(os.path.join(out_dir, 'predicted_eddies'), exist_ok=True)
    
    # Load SSH data using xarray for better handling
    ds = xr.open_dataset(nc_path)
    ssh_all = ds['ssh']  # Extract SSH variable
    lon = ds['lon'].values
    lat = ds['lat'].values
    
    for t in range(predictions.shape[0]):
        # Get SSH slice for this timestep
        ssh_slice = ssh_all.isel(time=t).values
        
        # Create figure with larger size
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 1. Land mask
        land_mask = ssh_slice < -0.25
        # 2. Brighter ocean, land in white
        cmap = cm.get_cmap('jet').copy()
        cmap.set_under('white')
        cmap.set_over('white')
        im = ax.pcolormesh(lon, lat, ssh_slice, cmap=cmap, vmin=-0.3, vmax=0.3, shading='gouraud')
        # 3. Black coastline
        ax.contour(lon, lat, land_mask, colors='black', linewidths=1)
        # 4. Colorbar
        cbar = plt.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.04)
        cbar.set_ticks([-0.3, 0, 0.3])
        cbar.set_label('SSH (m)', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        ax.tick_params(labelsize=10)

        # Set aspect ratio and turn off axes
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Get prediction mask for this timestep
        pred_mask = predictions[t].argmax(axis=-1)
        
        # === Region-wise land exclusion ===
        final_mask = np.zeros_like(pred_mask)
        for class_val in [1, 2]:  # cyclonic, anticyclonic
            labeled = measure.label(pred_mask == class_val)
            for region in measure.regionprops(labeled):
                if region.area < 5:
                    continue
                y, x = region.centroid
                lat_center = np.interp(y, np.arange(len(lat)), lat)
                lon_center = np.interp(x, np.arange(len(lon)), lon)
                radius_pix = np.sqrt(region.area / np.pi)
                lon_r = radius_pix * (lon[1] - lon[0])
                lat_r = radius_pix * (lat[1] - lat[0])
                angles = np.linspace(0, 2*np.pi, 360)
                circle_lon = lon_center + lon_r * np.cos(angles)
                circle_lat = lat_center + lat_r * np.sin(angles)
                on_land = False
                for clat, clon in zip(circle_lat, circle_lon):
                    idx_lat = np.argmin(np.abs(lat - clat))
                    idx_lon = np.argmin(np.abs(lon - clon))
                    if ssh_slice[idx_lat, idx_lon] < -0.25:
                        on_land = True
                        break
                if on_land:
                    continue
                final_mask[labeled == region.label] = class_val
        # Overlay eddy contours with smooth lines using final_mask
        if np.any(final_mask == 1):  # Cyclonic eddies
            ax.contour(lon, lat, final_mask == 1, colors='blue', linewidths=1.5, alpha=0.6)
        if np.any(final_mask == 2):  # Anticyclonic eddies
            ax.contour(lon, lat, final_mask == 2, colors='red', linewidths=1.5, alpha=0.6)
        
        # Set title and save
        ax.set_title(f"Detected Eddies – Day {t:04d}", fontsize=14)
        fn = os.path.join(out_dir, 'predicted_eddies', f"eddy_pred_{t:04d}.png")
        plt.tight_layout()
        plt.savefig(fn, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    # Close the dataset
    ds.close()

def save_circled_eddies(predictions, nc_path, out_dir, km_per_pixel=27.8):
    """
    Saves SSH+eddy-circles images to out_dir/circled_eddies.
    Red=anticyclonic, Blue=cyclonic.
    """
    os.makedirs(os.path.join(out_dir, 'circled_eddies'), exist_ok=True)
    
    # Load SSH data using netCDF4
    with nc.Dataset(nc_path, 'r') as nc_file:
        ssh_data_nc = nc_file.variables['ssh'][:]
        lat_nc = nc_file.variables['lat'][:]
        lon_nc = nc_file.variables['lon'][:]
    
    for t in range(predictions.shape[0]):
        pred_mask = np.argmax(predictions[t], axis=-1)
        ssh_frame_nc = ssh_data_nc[t]

        fig, ax = plt.subplots(figsize=(6, 5))

        # Colormap for SSH: viridis with land as white
        land_mask = ssh_frame_nc < -0.25
        cmap = cm.get_cmap('jet').copy()
        cmap.set_under('white')
        cmap.set_over('white')
        img = ax.pcolormesh(lon_nc, lat_nc, ssh_frame_nc, cmap=cmap, vmin=-0.3, vmax=0.3, shading='gouraud')
        ax.contour(lon_nc, lat_nc, land_mask, colors='black', linewidths=1)
        ax.set_aspect('equal', 'box')
        # Add colorbar with explicit ticks and labels
        cbar = plt.colorbar(img, ax=ax, extend='both', shrink=0.8, fraction=0.046, pad=0.04)
        cbar.set_label('SSH (m)', fontsize=12)
        cbar.set_ticks([-0.25, 0.0, 0.25])
        cbar.set_ticklabels(['-0.25', '0.00', '0.25'])
        cbar.ax.tick_params(labelsize=10)
        ax.tick_params(labelsize=10)

        for class_val, color in zip([1, 2], ['blue', 'red']):  # 1: cyclonic, 2: anticyclonic
            labeled = measure.label(pred_mask == class_val)

            for region in measure.regionprops(labeled):
                if region.area < 5:
                    continue

                y, x = region.centroid
                lat_center = np.interp(y, np.arange(len(lat_nc)), lat_nc)
                lon_center = np.interp(x, np.arange(len(lon_nc)), lon_nc)

                # Estimate radius in pixels and convert to lon/lat radius
                radius_pix = np.sqrt(region.area / np.pi)
                lon_res = lon_nc[1] - lon_nc[0]
                lat_res = lat_nc[1] - lat_nc[0]
                lon_r = radius_pix * lon_res
                lat_r = radius_pix * lat_res

                # Generate points on the circle (360 points)
                angles = np.linspace(0, 2*np.pi, 360)
                circle_lon = lon_center + lon_r * np.cos(angles)
                circle_lat = lat_center + lat_r * np.sin(angles)

                # Check if any circle point lies on land (SSH < -0.25)
                on_land = False
                for clat, clon in zip(circle_lat, circle_lon):
                    lat_idx = np.argmin(np.abs(lat_nc - clat))
                    lon_idx = np.argmin(np.abs(lon_nc - clon))
                    if ssh_frame_nc[lat_idx, lon_idx] < -0.25:
                        on_land = True
                        break

                if on_land:
                    continue  # Don't draw this eddy

                # Draw the eddy circle
                circ = Circle((lon_center, lat_center), lon_r, edgecolor=color, facecolor='none', linewidth=2)
                ax.add_patch(circ)

                # Draw the eddy center as a visible dot
                ax.plot(lon_center, lat_center, marker='o', color=color, markersize=4, markeredgecolor='black', markeredgewidth=0.5)

        ax.set_title(f"Eddies with Circles - Day {t:02d}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'circled_eddies', f"circled_eddy_{t:04d}.png"),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == "__main__":
    # ... existing training and evaluation code ...
    
    # Load NC data for visualization
    nc_path = os.path.join(DATA_DIR, 'ssh_3months_2020.nc')

    filtered_SSH_train_data = np.load(os.path.join(DATA_DIR, 'filtered_SSH_train_data.npy'))
    train_preds = eddydlv3net.predict(filtered_SSH_train_data, verbose=1)
    save_discrete_colormap_predictions(train_preds, nc_path, './eddydlv3net')
    save_circled_eddies(train_preds, nc_path, './eddydlv3net')

    # === New: Test data predictions ===
    test_data_path = os.path.join(DATA_DIR, 'filtered_SSH_test_data.npy')
    if os.path.exists(test_data_path):
        filtered_SSH_test_data = np.load(test_data_path)
        test_preds = eddydlv3net.predict(filtered_SSH_test_data, verbose=1)
        test_out_dir = './eddydlv3net_test'
        save_discrete_colormap_predictions(test_preds, nc_path, test_out_dir)
        save_circled_eddies(test_preds, nc_path, test_out_dir)
        print(f"✅ Test predictions saved in {test_out_dir}/predicted_eddies and {test_out_dir}/circled_eddies")
    else:
        print(f"Warning: Test data file not found at {test_data_path}")

    # === Eddy tracking data collection ===
    # Load SSH data using xarray for consistent handling
    ds = xr.open_dataset(nc_path)
    ssh_all = ds['ssh']
    lon = ds['lon'].values
    lat = ds['lat'].values
    
    eddy_days, eddy_labels, eddy_lat_centers, eddy_lon_centers, eddy_radii_km = [], [], [], [], []
    eddy_report_lines = []  # For saving formatted print statements
    km_per_pixel = 27.8

    for t in range(predictions.shape[0]):
        # Get SSH slice for this timestep
        ssh_slice = ssh_all.isel(time=t).values
        pred_mask = predictions[t].argmax(axis=-1)
        
        # Create land mask
        land_mask = ssh_slice < -0.25
        
        for class_val, label in [(1, "Cyclonic"), (2, "Anti-Cyclonic")]:
            labeled = measure.label(pred_mask == class_val)
            for region in measure.regionprops(labeled):
                if region.area < 5:
                    continue
                
                # Get eddy center
                y0, x0 = region.centroid
                y_idx, x_idx = int(round(y0)), int(round(x0))
                
                # Skip if on land
                if land_mask[y_idx, x_idx]:
                    continue
                
                # Convert pixel coordinates to lat/lon
                lon_center = np.interp(x0, np.arange(len(lon)), lon)
                lat_center = np.interp(y0, np.arange(len(lat)), lat)
                
                # Compute radius
                radius = np.sqrt(region.area / np.pi)
                radius_km = radius * km_per_pixel
                
                # Store eddy data
                print(f"[Day {t:02d}] {label} Eddy: Center=({lat_center:.2f}°, {lon_center:.2f}°), Radius={radius_km:.1f} km")
                line = f"[Day {t:02d}] {label} Eddy: Center=({lat_center:.2f}°, {lon_center:.2f}°), Radius={radius_km:.1f} km"
                eddy_report_lines.append(line)
                
                eddy_days.append(t)
                eddy_labels.append(label)
                eddy_lat_centers.append(lat_center)
                eddy_lon_centers.append(lon_center)
                eddy_radii_km.append(radius_km)
    
    # Close the dataset
    ds.close()

    # Save eddy tracking data to NetCDF
    output_nc_file = r'C:\Users\jalla\Downloads\Eddy\Eddy_Detection_Tracking\detection_method\eddy_tracking_data.nc'
    with nc.Dataset(output_nc_file, 'w', format='NETCDF4') as ds:
        # Global attributes
        ds.description = 'Eddy tracking data from model predictions.'
        ds.source = 'FrameWork.py'
        ds.training_data_shape = str(SSH_train.shape)
        ds.ground_truth_shape = str(Seg_train.shape)

        # Dimensions
        num_eddies = len(eddy_days)
        if num_eddies > 0:
            ds.createDimension('eddy', num_eddies)

            # Variables
            days = ds.createVariable('day', 'i4', ('eddy',))
            labels = ds.createVariable('label', str, ('eddy',))
            lats = ds.createVariable('latitude', 'f4', ('eddy',))
            lons = ds.createVariable('longitude', 'f4', ('eddy',))
            radii = ds.createVariable('radius_km', 'f4', ('eddy',))
            report_lines = ds.createVariable('report', str, ('eddy',))

            # Add data to variables
            days[:] = eddy_days
            labels[:] = np.array(eddy_labels)
            lats[:] = eddy_lat_centers
            lons[:] = eddy_lon_centers
            radii[:] = eddy_radii_km
            report_lines[:] = np.array([s.encode('utf-8') for s in eddy_report_lines])  # UTF-8 encoding
    
    print(f"\nEddy data saved to {output_nc_file}")


print("Training samples found:", SSH_train.shape[0])

data = np.load('C:/Users/jalla/Downloads/Eddy/Sea/filtered_SSH_train_data.npy')
print(data.shape)