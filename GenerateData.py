import os
import numpy as np
import HelpFunc as hf
import argparse
from sklearn.utils import shuffle

def genarate_data(Type='train', DatasetPath="dataset/",
                  input_size=33, output_size=21, chanels=3):
    # images_path = os.path.join(DatasetPath, Type, "*.png")
    images_path = os.path.join(DatasetPath, Type, "*.bmp")
    list_images_path = hf.sorted_list(images_path)

    if os.path.exists(os.path.join(DatasetPath, f"data_{Type}.npy")):
        return

    print(f"\n==================== Generating data {Type} ====================")
    stride = output_size - 1
    scale = 3
    padding = (input_size - output_size) // 2
    data = np.zeros((0, input_size, input_size, chanels))
    labels = np.zeros((0, output_size, output_size, chanels))
    for impath in list_images_path:
        print(impath)
        img_label = hf.read_image(impath)
        img_label = hf.im2double(img_label)

        im_input = hf.read_image(impath)
        im_input = hf.make_lr(im_input, scale=scale)
        im_input = hf.im2double(im_input)

        h = img_label.shape[0]
        w = img_label.shape[1]

        for x in np.arange(start=0, stop=h-input_size, step=stride):
            for y in np.arange(start=0, stop=w-input_size, step=stride):
                subim_input = im_input[x: x + input_size, y: y + input_size]
                subim_label = img_label[x + padding: x + padding + output_size,
                                        y + padding: y + padding + output_size]
                data = np.vstack([data, [subim_input]])
                labels = np.vstack([labels, [subim_label]])

    data, labels = shuffle(data, labels)
    np.save(os.path.join(DatasetPath, f"data_{Type}.npy"), data)
    np.save(os.path.join(DatasetPath, f"labels_{Type}.npy"), labels)
    print(f"=================== End generating data {Type} ===================")



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default="dataset/", help='-')
parser.add_argument('--input_size', type=int, default=33, help='-')
parser.add_argument('--chanels', type=int, default=3, help='-')

FLAGS, unparsed = parser.parse_known_args()

InputSize = FLAGS.input_size
OutputSze = InputSize - 12
DatasetPath = FLAGS.dataset_path
Chanels = FLAGS.chanels

genarate_data(Type='train', DatasetPath=DatasetPath,
              input_size=InputSize, output_size=OutputSze, chanels=Chanels)

genarate_data(Type='validation', DatasetPath=DatasetPath,
              input_size=InputSize, output_size=OutputSze, chanels=Chanels)

genarate_data(Type='test', DatasetPath=DatasetPath,
              input_size=InputSize, output_size=OutputSze, chanels=Chanels)
