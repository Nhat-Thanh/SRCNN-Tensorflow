from Model import SRCNN
import HelpFunc as hf
import numpy as np
import argparse
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=float, default=2, help='-')

FLAGS, unparsed = parser.parse_known_args()

if FLAGS.scale < 1 or FLAGS.scale > 5:
    print("Invalid argument")
    exit()


print("\n\n==================================================",
      " Running ",
      "==================================================\n")

hf.makedir("demo/")
hf.makedir("demo/images")
hf.makedir("demo/results")
hf.makedir("demo/results/recon")
hf.makedir("demo/results/bicubic")
hf.makedir("demo/results/smooth")

model = SRCNN()
model.load_weights("checkpoint/model-ckpt")

list_image_names = os.listdir("demo/images/")

for imname in list_image_names:
    image = hf.read_image(f"demo/images/{imname}")
    bicubic_img = hf.upscale(image, FLAGS.scale)
    img_p12 = hf.plus12(bicubic_img)
    img_p12[6:-6, 6:-6] = bicubic_img
    image = hf.im2double(img_p12)
    image = np.expand_dims(image, axis=0)

    recon_img = model.predict(image, verbose=1)

    recon_img = hf.double2uint8(recon_img)
    recon_img = cv2.cvtColor(recon_img, cv2.COLOR_YCR_CB2BGR)
    bicubic_img = cv2.cvtColor(bicubic_img, cv2.COLOR_YCR_CB2BGR)
    smooth_img = cv2.GaussianBlur(bicubic_img, (3, 3), 0)

    cv2.imwrite(f"demo/results/recon/{os.path.splitext(imname)[0]}-recon.bmp", recon_img)
    cv2.imwrite(f"demo/results/bicubic/{os.path.splitext(imname)[0]}-bicubic.bmp", bicubic_img)
    cv2.imwrite(f"demo/results/smooth/{os.path.splitext(imname)[0]}-smooth.bmp", smooth_img)
