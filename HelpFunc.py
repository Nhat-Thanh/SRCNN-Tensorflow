import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import pandas as pd


def resize_cubic(src, dsize):
    img = cv2.resize(src, dsize, interpolation=cv2.INTER_CUBIC)
    return img


def read_image(path):
    bgr_img = cv2.imread(path, cv2.IMREAD_COLOR)
    ycbcr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCR_CB)
    return ycbcr_img


def upscale(src, scale):
    h = int(src.shape[0] * scale)
    w = int(src.shape[1] * scale)
    upscaled_img = resize_cubic(src, (w, h)) 
    upscaled_img = cv2.GaussianBlur(upscaled_img, (3, 3), sigmaX=0.2)
    return upscaled_img


def plus12(src):
    h = src.shape[0] + 12
    w = src.shape[1] + 12
    img = resize_cubic(src, (w, h))
    return img


def im2double(src):
    return src / 255


def make_lr(src, scale):
    h = src.shape[0]
    w = src.shape[1]
    blur_img = cv2.GaussianBlur(src, (3, 3), sigmaX=0.7)
    img_lr = resize_cubic(blur_img, (w // scale, h // scale))
    img_lr = resize_cubic(img_lr, (w, h))
    return img_lr


def sorted_list(path):
    tmplist = glob.glob(path)
    tmplist.sort()
    return tmplist


def makedir(path):
    try:
        os.mkdir(path)
    except:
        pass


def double2uint8(img):
    result = np.squeeze(img, axis=0)
    result = result * 255
    result = result.astype("uint8")
    return result


def PSNR(y_true, y_pred, max_val=1.0):
    return 10 * tf.experimental.numpy.log10(
        max_val * max_val / tf.reduce_mean(
            tf.square(y_pred - y_true)
        )
    )


def AppendHistory(df):
    if os.path.exists("history.csv"):
        print("\n\nhistory file exists\n")
        history = pd.read_csv("history.csv")
        loss = np.array(history[['loss']].values, dtype=np.float32)
        PSNR = np.array(history[['PSNR']].values, dtype=np.float32)
        val_loss = np.array(history[['val_loss']].values, dtype=np.float32)
        val_PSNR = np.array(history[['val_PSNR']].values, dtype=np.float32)
    else:
        print("\n\nhistory file does not exist\n")
        loss = np.array([], dtype=np.float32)
        PSNR = np.array([], dtype=np.float32)
        val_loss = np.array([], dtype=np.float32)
        val_PSNR = np.array([], dtype=np.float32)

    loss = np.append(loss, df[['loss']].values)
    PSNR = np.append(PSNR, df[['PSNR']].values)
    val_loss = np.append(val_loss, df[['val_loss']].values)
    val_PSNR = np.append(val_PSNR, df[['val_PSNR']].values)

    table = pd.DataFrame({
        "loss": loss,
        "PSNR": PSNR,
        "val_loss": val_loss,
        "val_PSNR": val_PSNR
    })

    return table
