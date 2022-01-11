import tensorflow as tf
from Model import SRCNN
import pandas as pd
import HelpFunc as hf
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=2500, help='-')
parser.add_argument('--batch_size', type=int, default=128, help='-')

FLAGS, unparsed = parser.parse_known_args()

X_tr = np.load("dataset/data_train.npy")
Y_tr = np.load("dataset/labels_train.npy")

X_val = np.load("dataset/data_validation.npy")
Y_val = np.load("dataset/labels_validation.npy")

X_te = np.load("dataset/data_test.npy")
Y_te = np.load("dataset/labels_test.npy")

model = SRCNN()

ckpt_path = "checkpoint/model-ckpt"
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    save_weights_only=True,
    verbose=1
)

if os.path.exists(f"{ckpt_path}.index"):
    print(f"\n\nLoad Model from \"{ckpt_path}\"")
    model.load_weights(ckpt_path)
else:
    print(f"\n\n\"{ckpt_path}\" DO NOT EXIST")


if os.path.exists("checkpoint/cur_epoch.npy"):
    cur_epoch = np.load("checkpoint/cur_epoch.npy")
else:
    cur_epoch = np.int32(0)
print(f"\n\nCurrent epoch is {cur_epoch}\n")


print("\n\n==================================================",
      " Train ",
      "==================================================\n")

history = model.fit(
    x=X_tr, y=Y_tr,
    batch_size=FLAGS.batch_size,
    epochs=FLAGS.epoch,
    validation_data=(X_val, Y_val),
    callbacks=[ckpt_callback]
)

print("\n\n==================================================",
      " Evaluate ",
      "==================================================\n")

model.evaluate(X_te, Y_te)

df = pd.DataFrame(history.history)
df = hf.AppendHistory(df)
df.to_csv("history.csv", index=False)

cur_epoch = cur_epoch + FLAGS.epoch
print(f"\nTotal trained epochs is {cur_epoch}\n")

np.save("checkpoint/cur_epoch.npy", cur_epoch)
