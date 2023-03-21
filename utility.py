import numpy as np
import imageio as io
import tensorflow as tf
import os
from keras import backend as K


def createAndSaveImage(image_to_save, output_name):
    io.imwrite(output_name, image_to_save)
    return True

def applyMask(input, mask):
    masked = input
    masked[np.where(mask == 0)] = 0
    return masked

def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,
                                  shuffle_size=10000, seed=42):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=seed)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

def getAllFiles(dir, result = None):
    if result is None:
        result = []
    for entry in os.listdir(dir):
        entrypath = os.path.join(dir, entry)
        if os.path.isdir(entrypath):
            getAllFiles(entrypath ,result)
        else:
            result.append(entrypath)
    result = sorted(result)
    return result

def load_to_numpy(path):
    image = io.imread(path)
    return np.array(image)

def dice_coef(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred, 1)