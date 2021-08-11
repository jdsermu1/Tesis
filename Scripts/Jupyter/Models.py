
##

import os, gc, glob, math, cv2, random
import numpy as np
import pandas as pd
from datetime import datetime
from progressbar import ProgressBar
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Dense, LeakyReLU, Reshape, InputSpec, Dropout, Flatten, \
    LeakyReLU, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers.schedules import ExponentialDecay, InverseTimeDecay
from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, TerminateOnNaN, TensorBoard
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.utils import Sequence
from imblearn.over_sampling import RandomOverSampler
from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

##

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

mixed_precision.set_global_policy('mixed_float16')
# tf.config.experimental.enable_tensor_float_32_execution(False)
# tf.config.gpu.set_per_process_memory_fraction(0.87)
# tf.config.gpu.set_per_process_memory_growth(True)



##

db_folder_kaggle = os.path.join(os.getcwd(), '../DR Databases/Kaggle')
# Dependiendo de la estrategia a probar cambiar nombre carpeta
strategy = "strategy 1"
images_folder = os.path.join(db_folder_kaggle, strategy)
all_images_folder = os.path.join(db_folder_kaggle, "preprocessed images")
# ------------------------------------------------------------
labels_path = os.path.join(db_folder_kaggle, "labels.csv")
train_images_folder = os.path.join(images_folder, "train")
validation_images_folder = os.path.join(images_folder, "validation")
test_images_folder = os.path.join(images_folder, "test")
log_dir = os.path.join(db_folder_kaggle, "logs")
current_time = datetime.now()


##

class FractionalPooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_ratio=None, pseudo_random=False, overlap=False, name='FractionPooling2D', **kwargs):
        self.pool_ratio = pool_ratio
        self.input_spec = InputSpec(ndim=4)
        self.pseudo_random = pseudo_random
        self.overlap = overlap
        super(FractionalPooling2D, self).__init__(**kwargs)

    def call(self, input):
        batch_tensor, _, _ = tf.nn.fractional_max_pool(input, pooling_ratio=self.pool_ratio,
                                                       pseudo_random=self.pseudo_random,
                                                       overlapping=self.overlap)
        return batch_tensor

    def compute_output_shape(self, input_shape):
        if (K.image_dim_ordering() == 'channels_last' or K.image_dim_ordering() == 'tf'):
            if (input_shape[0] != None):
                batch_size = int(input_shape[0] / self.pool_ratio[0])
            else:
                batch_size = input_shape[0]
            width = int(input_shape[1] / self.pool_ratio[1])
            height = int(input_shape[2] / self.pool_ratio[2])
            channels = int(input_shape[3] / self.pool_ratio[3])
            return (batch_size, width, height, channels)

        elif (K.image_dim_ordering() == 'channels_first' or K.image_dim_ordering() == 'th'):
            if (input_shape[0] != None):
                batch_size = int(input_shape[0] / self.pool_ratio[0])
            else:
                batch_size = input_shape[0]
            channels = int(input_shape[1] / self.pool_ratio[1])
            width = int(input_shape[2] / self.pool_ratio[2])
            height = int(input_shape[3] / self.pool_ratio[3])
            return (batch_size, channels, width, height)

    def get_config(self):
        config = {'pooling_ratio': self.pool_ratio, 'pseudo_random': self.pseudo_random, 'overlap': self.overlap,
                  'name': self.name}
        base_config = super(FractionalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)



##

def model_A(kernel_max_norm=0.1, input_shape=(540, 540, 3)):
    initializer = None  # HeUniform()
    constraint = None  # MaxNorm(max_value=kernel_max_norm, axis=[0,1,2])
    constraint_bias = None
    fmp_type = 'float32'
    fmp_overlap = True
    fmp_pesudo_random = True
    conv_padding = 'same'
    pool_ratio = (1, 1.8, 1.8, 1)
    leakyr_alpha = 0.333

    m = Sequential()

    m.add(Conv2D(32, (5, 5), input_shape=input_shape, padding=conv_padding, kernel_initializer=initializer,
                 kernel_constraint=constraint, bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    m.add(FractionalPooling2D(pool_ratio=pool_ratio, pseudo_random=fmp_pesudo_random, overlap=fmp_overlap,
                              dtype=fmp_type))

    m.add(Conv2D(64, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    m.add(FractionalPooling2D(pool_ratio=pool_ratio, pseudo_random=fmp_pesudo_random, overlap=fmp_overlap,
                              dtype=fmp_type))

    m.add(Conv2D(96, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    m.add(FractionalPooling2D(pool_ratio=pool_ratio, pseudo_random=fmp_pesudo_random, overlap=fmp_overlap,
                              dtype=fmp_type))

    m.add(Conv2D(128, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    m.add(FractionalPooling2D(pool_ratio=pool_ratio, pseudo_random=fmp_pesudo_random, overlap=fmp_overlap,
                              dtype=fmp_type))

    m.add(Conv2D(160, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    m.add(FractionalPooling2D(pool_ratio=pool_ratio, pseudo_random=fmp_pesudo_random, overlap=fmp_overlap,
                              dtype=fmp_type))

    m.add(Conv2D(192, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    m.add(FractionalPooling2D(pool_ratio=pool_ratio, pseudo_random=fmp_pesudo_random, overlap=fmp_overlap,
                              dtype=fmp_type))

    m.add(Conv2D(224, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    m.add(FractionalPooling2D(pool_ratio=pool_ratio, pseudo_random=fmp_pesudo_random, overlap=fmp_overlap,
                              dtype=fmp_type))

    m.add(Conv2D(256, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    m.add(Dropout(32.0 / 352))
    m.add(FractionalPooling2D(pool_ratio=pool_ratio, pseudo_random=fmp_pesudo_random, overlap=fmp_overlap,
                              dtype=fmp_type))

    m.add(Conv2D(288, (2, 2), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    m.add(Dropout(32.0 / 384))
    m.add(FractionalPooling2D(pool_ratio=pool_ratio, pseudo_random=fmp_pesudo_random, overlap=fmp_overlap,
                              dtype=fmp_type))

    m.add(Conv2D(320, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    m.add(Dropout(64.0 / 416))

    m.add(Conv2D(356, (1, 1), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    l = Dropout(64.0 / 448)
    m.add(l)

    m.add(Conv2D(5, l.output_shape[1:3], kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(Softmax(dtype='float32'))
    m.add(Flatten())
    m.summary()
    return m


def model_B(kernel_max_norm=0.1):
    initializer = None  # HeUniform()
    constraint = None  # MaxNorm(max_value=kernel_max_norm, axis=[0,1,2])
    constraint_bias = None

    m = Sequential()
    m.add(Conv2D(32, (5, 5), input_shape=(540, 540, 3), padding='same', kernel_initializer=initializer,
                 kernel_constraint=constraint, bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.5, 1.5, 1), pseudo_random=True, overlap=True))

    m.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.5, 1.5, 1), pseudo_random=True, overlap=True, dtype='float32'))

    m.add(Conv2D(96, (3, 3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.5, 1.5, 1), pseudo_random=True, overlap=True, dtype='float32'))

    m.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.5, 1.5, 1), pseudo_random=True, overlap=True, dtype='float32'))

    m.add(Conv2D(160, (3, 3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.5, 1.5, 1), pseudo_random=True, overlap=True, dtype='float32'))

    m.add(Conv2D(192, (3, 3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.5, 1.5, 1), pseudo_random=True, overlap=True, dtype='float32'))

    m.add(Conv2D(224, (3, 3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.5, 1.5, 1), pseudo_random=True, overlap=True, dtype='float32'))

    m.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.5, 1.5, 1), pseudo_random=True, overlap=True, dtype='float32'))

    m.add(Conv2D(288, (3, 3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.5, 1.5, 1), pseudo_random=True, overlap=True, dtype='float32'))

    m.add(Conv2D(320, (3, 3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.5, 1.5, 1), pseudo_random=True, overlap=True, dtype='float32'))

    m.add(Conv2D(352, (3, 3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.6, 1.6, 1), pseudo_random=True, overlap=True, dtype='float32'))
    m.add(Dropout(32.0 / 352))

    m.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(FractionalPooling2D(pool_ratio=(1, 1.5, 1.5, 1), pseudo_random=True, overlap=True, dtype='float32'))
    m.add(Dropout(32.0 / 384))

    m.add(Conv2D(416, (2, 2), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    m.add(Dropout(64.0 / 416))

    m.add(Conv2D(448, (1, 1), padding='same', kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=0.333))
    l = Dropout(64.0 / 448)
    m.add(l)

    m.add(Conv2D(5, l.output_shape[1:3], kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(Softmax(dtype='float32'))
    m.add(Flatten())
    m.summary()
    return m


def model_A_modified(kernel_max_norm=0.1, input_shape=(540, 540, 3)):
    initializer = None  # HeUniform()
    constraint = None  # MaxNorm(max_value=kernel_max_norm, axis=[0,1,2])
    constraint_bias = None
    fmp_type = 'float32'
    fmp_overlap = True
    fmp_pesudo_random = True
    conv_padding = 'same'
    pool_ratio = (1, 1.8, 1.8, 1)
    leakyr_alpha = 0.333

    m = Sequential()

    m.add(Conv2D(32, (5, 5), input_shape=input_shape, padding=conv_padding, kernel_initializer=initializer,
                 kernel_constraint=constraint, bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    #     m.add(FractionalPooling2D(pool_ratio=pool_ratio,pseudo_random = fmp_pesudo_random,overlap=fmp_overlap, dtype=fmp_type))
    m.add(MaxPool2D())

    m.add(Conv2D(64, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    #     m.add(FractionalPooling2D(pool_ratio=pool_ratio,pseudo_random = fmp_pesudo_random,overlap=fmp_overlap, dtype=fmp_type))
    m.add(MaxPool2D())

    m.add(Conv2D(96, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    #     m.add(FractionalPooling2D(pool_ratio=pool_ratio,pseudo_random = fmp_pesudo_random,overlap=fmp_overlap, dtype=fmp_type))
    m.add(MaxPool2D())

    m.add(Conv2D(128, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    #     m.add(FractionalPooling2D(pool_ratio=pool_ratio,pseudo_random = fmp_pesudo_random,overlap=fmp_overlap, dtype=fmp_type))
    m.add(MaxPool2D())

    m.add(Conv2D(160, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    #     m.add(FractionalPooling2D(pool_ratio=pool_ratio,pseudo_random = fmp_pesudo_random,overlap=fmp_overlap, dtype=fmp_type))
    m.add(MaxPool2D())

    m.add(Conv2D(192, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    #     m.add(FractionalPooling2D(pool_ratio=pool_ratio,pseudo_random = fmp_pesudo_random,overlap=fmp_overlap, dtype=fmp_type))
    m.add(MaxPool2D())

    m.add(Conv2D(224, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    #     m.add(FractionalPooling2D(pool_ratio=pool_ratio,pseudo_random = fmp_pesudo_random,overlap=fmp_overlap, dtype=fmp_type))
    m.add(MaxPool2D())

    m.add(Conv2D(256, (3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    m.add(Dropout(32.0 / 352))
    #     m.add(FractionalPooling2D(pool_ratio=pool_ratio,pseudo_random = fmp_pesudo_random,overlap=fmp_overlap, dtype=fmp_type))
    m.add(MaxPool2D())

    #     m.add(Conv2D(288,(2, 2), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint, bias_constraint=constraint_bias))
    #     m.add(LeakyReLU(alpha=leakyr_alpha))
    #     m.add(Dropout(32.0/384))
    # #     m.add(FractionalPooling2D(pool_ratio=pool_ratio,pseudo_random = fmp_pesudo_random,overlap=fmp_overlap, dtype=fmp_type))
    #     m.add(MaxPool2D())

    #     m.add(Conv2D(320,(3, 3), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint, bias_constraint=constraint_bias))
    #     m.add(LeakyReLU(alpha=leakyr_alpha))
    #     m.add(Dropout(64.0/416))

    m.add(Conv2D(356, (1, 1), padding=conv_padding, kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(LeakyReLU(alpha=leakyr_alpha))
    l = Dropout(64.0 / 448)
    m.add(l)

    m.add(Conv2D(5, l.output_shape[1:3], kernel_initializer=initializer, kernel_constraint=constraint,
                 bias_constraint=constraint_bias))
    m.add(Softmax(dtype='float32'))
    m.add(Flatten())
    m.summary()
    return m


def model_D(input_shape):
    m = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape, pooling="max")
    m.get_layer("block5_conv3").trainable = False
    last_layer = m.get_layer("block5_pool")
    x = Flatten()(last_layer.output)
    x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)
    x = Dense(5, activation="softmax")(x)
    m = Model(m.input, x)
    m.summary()
    return m


##

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_accuracy') > 0.8476):
            print("Reached reported accuracy so cancelling training!")
            self.model.stop_training = True

    def on_test_batch_begin(self, epoch, logs={}):
        gc.collect()


def scheduler(epoch, lr):
    return lr * math.exp(-0.05)


def schedulerD(epoch, lr):
    if epoch % 5 == 0:
        return math.pow(lr, 15)
    else:
        return lr


##

def load_data_generator(batch_size, img_dims):
    train_generator = ImageDataGenerator(rotation_range=360, shear_range=.2, rescale=1.0 / 255.0)
    train_flow = train_generator.flow_from_directory(train_images_folder,
                                                     batch_size=batch_size,
                                                     class_mode="categorical",
                                                     target_size=img_dims)
    validation_generator = ImageDataGenerator(rescale=1.0 / 255.0)
    validation_flow = validation_generator.flow_from_directory(validation_images_folder,
                                                               batch_size=batch_size,
                                                               class_mode="categorical",
                                                               target_size=img_dims)
    data_generator = {"train": train_flow, "validation": validation_flow}
    return data_generator


rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.0)
rotation = tf.keras.layers.experimental.preprocessing.RandomRotation((0, 1), fill_mode='nearest')
flip = tf.keras.layers.experimental.preprocessing.RandomFlip()


def load_data_using_tfdata(folders, batch_size, img_dims, caches, parallels):
    def parse_image(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        class_names = np.array(os.listdir(images_folder + '/train'))
        label = parts[-2] == class_names
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [img_dims[0], img_dims[1]])
        img = rescaling(img)
        return img, label

    def data_augmentation(x, y):
        x = rotation(x)
        x = flip(x)
        return x, y

    def prepare_for_training(ds, cache=f'./{strategy}.cache', shuffle_buffer_size=200, folder_name="train",
                             parallel=tf.data.AUTOTUNE):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size).repeat().batch(batch_size)
        if folder_name == "train":
            ds = ds.map(data_augmentation, num_parallel_calls=parallel)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    data_generator = {}
    for i, x in enumerate(folders):
        dir_extend = images_folder + '/' + x
        list_ds = tf.data.Dataset.list_files(glob.glob(os.path.join(dir_extend, "**", '*.jpeg')))
        labeled_ds = list_ds.map(parse_image, num_parallel_calls=parallels[i])
        data_generator[x] = prepare_for_training(labeled_ds, cache=caches[i], folder_name=x, parallel=parallels[i])
    return data_generator


def load_data(validation_size, test_size, batch_size=32, img_dims=(540, 540),
              caches={"train": False, "test": False, "validation": False}, parallel=5):
    assert (0 < validation_size < 1 and 0 < test_size < 1 and validation_size + test_size < 1)

    def parse_image(img_name, y):
        file_path = tf.strings.join([all_images_folder + os.path.sep, img_name, ".jpeg"])
        label = tf.one_hot(y, 5)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [img_dims[0], img_dims[1]])
        img = rescaling(img)
        return img, label

    def data_augmentation(x, y):
        x = rotation(x)
        x = flip(x)
        return x, y

    def prepare_for_training(ds, folder_name, shuffle_buffer_size=200):
        if caches[folder_name]:
            if isinstance(caches[folder_name], str):
                ds = ds.cache(caches[folder_name])
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        if folder_name == "train":
            ds = ds.map(data_augmentation, num_parallel_calls=parallel)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    data_df = pd.read_csv(labels_path)
    temp_df, data_test_df = train_test_split(data_df, test_size=test_size, stratify=data_df["level"])
    data_train_df, data_validation_df = train_test_split(temp_df, test_size=validation_size / (1 - test_size),
                                                         stratify=temp_df["level"])
    data_dfs = {"train": data_train_df, "test": data_test_df, "validation": data_validation_df}
    data_dfs_len = {"train": len(data_train_df), "test": len(data_test_df), "validation": len(data_validation_df)}
    print(data_df.groupby("level").size(), data_test_df.groupby("level").size(), data_train_df.groupby("level").size(),
          data_validation_df.groupby("level").size())

    data_generator = {}

    for set_name, set_df in data_dfs.items():
        set_ds = tf.data.Dataset.from_tensor_slices((tf.cast(set_df["image"].values, tf.string),
                                                     tf.cast(set_df['level'].values, tf.int32)))
        labeled_ds = set_ds.map(parse_image, num_parallel_calls=parallel)
        data_generator[set_name] = prepare_for_training(labeled_ds, folder_name=set_name)

    return data_generator


def load_data_balanced_test_val(validation_size, test_size, train_size=None, balance=False, batch_size=32,
                                img_dims=(540, 540),
                                caches={"train": False, "test": False, "validation": False}, parallel=5):
    def parse_image(img_name, y):
        file_path = tf.strings.join([all_images_folder + os.path.sep, img_name, ".jpeg"])
        label = tf.one_hot(y, 5)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [img_dims[0], img_dims[1]], method='nearest')
        img = rescaling(img)
        return img, label

    def data_augmentation(x, y):
        x = rotation(x)
        x = flip(x)
        return x, y

    def prepare_for_training(ds, folder_name, shuffle_buffer_size=200):
        if caches[folder_name]:
            if isinstance(caches[folder_name], str):
                ds = ds.cache(caches[folder_name])
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        if folder_name == "train":
            ds = ds.map(data_augmentation, num_parallel_calls=parallel)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    data_df = pd.read_csv(labels_path)
    data_train_df, data_test_df, data_validation_df = None, None, None

    for label in range(5):
        data_label_df = data_df[data_df["level"] == label]
        temp_df, data_label_test_df = train_test_split(data_label_df, test_size=test_size)
        data_label_train_df, data_label_validation_df = train_test_split(temp_df, test_size=validation_size)
        if train_size and len(data_label_train_df) > train_size:
            data_label_train_df, _ = train_test_split(data_label_train_df, train_size=train_size)

        if data_test_df is None:
            data_test_df = data_label_test_df
        else:
            data_test_df = pd.concat([data_test_df, data_label_test_df])
        if data_validation_df is None:
            data_validation_df = data_label_validation_df
        else:
            data_validation_df = pd.concat([data_validation_df, data_label_validation_df])
        if data_train_df is None:
            data_train_df = data_label_train_df
        else:
            data_train_df = pd.concat([data_train_df, data_label_train_df])

    if balance:
        ros = RandomOverSampler()
        data_train_df, _ = ros.fit_resample(data_train_df, data_train_df["level"])

    data_dfs = {"train": data_train_df, "test": data_test_df, "validation": data_validation_df}
    data_dfs_len = {"train": len(data_train_df), "test": len(data_test_df), "validation": len(data_validation_df)}

    data_generator = {}

    for set_name, set_df in data_dfs.items():
        set_ds = tf.data.Dataset.from_tensor_slices((tf.cast(set_df["image"].values, tf.string),
                                                     tf.cast(set_df['level'].values, tf.int32)))
        labeled_ds = set_ds.map(parse_image, num_parallel_calls=parallel)
        data_generator[set_name] = prepare_for_training(labeled_ds, folder_name=set_name)

    return data_generator, data_dfs_len


##

log_dir_training = os.path.join(log_dir, "log_" + current_time.strftime("%Y-%m-%d-%H:%M:%S"))

callbacks = []
callbacks.append(MyCallback())  # Al alcanzar acc estimado del paper
callbacks.append(EarlyStopping(monitor="val_accuracy", patience=30, restore_best_weights=True))  # En caso de estancarse
# callbacks.append(LearningRateScheduler(schedulerD)) # Para prgramar funcion actualizarcion de tasa aprendizaje
callbacks.append(TerminateOnNaN())  # Terminar en caso fallas aprendizaje
callbacks.append(TensorBoard(log_dir=log_dir_training, histogram_freq=1))  # Utilizar Tensorboard

batch_size = 60
input_size = (224, 224, 3)
# data_generator = load_data_using_tfdata(["train", "validation"], 
#                                         batch_size, input_size[:-1], 
#                                         [False, False], 
#                                         [5, 5])
# f"{strategy}.dump"


data_generators, data_generators_dimensions = load_data_balanced_test_val(312, 100, train_size=25000, balance=True,
                                                                          batch_size=batch_size,
                                                                          img_dims=input_size[:-1],
                                                                          parallel=5)
print("Dimensions", data_generators_dimensions)

model = model_D(input_size)

model.compile(loss="categorical_crossentropy",
              metrics=["accuracy", Precision(), Recall()],
              optimizer=Adam(learning_rate=0.00001))  # , momentum=0.999))

history = model.fit(data_generators['train'],
                    validation_data=data_generators['validation'],
                    steps_per_epoch=data_generators_dimensions['train'] // batch_size,
                    validation_steps=data_generators_dimensions['validation'] // batch_size,
                    epochs=30, callbacks=callbacks)  # ,

##

# class_weight = {0:0.2668 ,1: 2.9911 ,2:1.3601, 3:10.3450, 4:11.5366}
# Strategy 1
# num_images_train = 80847
# num_images_validate = 500
# Strategy 2
# num_images_train = 7510
# num_images_validate = 500
# history = model.fit(data_generator['train'], 
#                     validation_data=data_generator['validation'],
#                     steps_per_epoch=num_images_train//batch_size, 
#                     validation_steps=num_images_validate//batch_size, 
#                     epochs=30, callbacks=callbacks)#, 
# class_weight=class_weight)
# history_df = pd.DataFrame(history.history)