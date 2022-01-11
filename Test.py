from Model import SRCNN
import HelpFunc as hf
import numpy as np
import os
import cv2

print("\n\n==================================================",
      " Test ",
      "==================================================\n")

cur_epoch = np.load("checkpoint/cur_epoch.npy")
hf.makedir("test/")

img_dir = f"test/{cur_epoch}"
recon_dir = f"{img_dir}/recon" 
bicubic_dir = f"{img_dir}/bicubic" 
smooth_dir = f"{img_dir}/smooth" 

hf.makedir(img_dir)
hf.makedir(recon_dir)
hf.makedir(bicubic_dir)
hf.makedir(smooth_dir)

model = SRCNN()
model.load_weights("checkpoint/model-ckpt")

# images_path = os.path.join("dataset", "test", "*.png")
images_path = os.path.join("dataset", "test", "*.bmp")
# labels_path = os.path.join("dataset", "testx2", "*.png")
labels_path = os.path.join("dataset", "testx2", "*.bmp")

list_images_path = hf.sorted_list(images_path)
list_labels_path = hf.sorted_list(labels_path)

for i in range(len(list_images_path)):
    impath = list_labels_path[i]
    img_hr = hf.read_image(impath)
    img_hr = hf.im2double(img_hr)
    img_hr = np.expand_dims(img_hr, axis=0)

    impath = list_images_path[i]
    img_lr = hf.read_image(impath)

    bicubic_img = hf.upscale(img_lr, 2)
    img_tmp = hf.im2double(bicubic_img)
    img_tmp = np.expand_dims(img_tmp, axis=0)
    bicubic_psnr = hf.PSNR(img_hr, img_tmp)

    smooth_img = cv2.GaussianBlur(bicubic_img, (3, 3), sigmaX=0.0)
    img_tmp = hf.im2double(smooth_img)
    img_tmp = np.expand_dims(img_tmp, axis=0)
    smooth_psnr = hf.PSNR(img_hr, img_tmp)

    img_p12 = hf.plus12(bicubic_img)
    img_p12[6:-6, 6:-6] = bicubic_img 
    img_lr = hf.im2double(img_p12)
    img_lr = np.expand_dims(img_lr, axis=0)

    recon_img = model.predict(img_lr, verbose=1)

    loss, psnr = model.evaluate(img_lr, img_hr)

    recon_img = hf.double2uint8(recon_img)
    recon_img = cv2.cvtColor(recon_img, cv2.COLOR_YCR_CB2BGR)
    bicubic_img = cv2.cvtColor(bicubic_img, cv2.COLOR_YCR_CB2BGR)
    smooth_img = cv2.cvtColor(smooth_img, cv2.COLOR_YCR_CB2BGR)

    recon_img_path = "{}/{:02d}-recon-psnr({:0.2f}).bmp".format(recon_dir, i, psnr)
    cv2.imwrite(recon_img_path, recon_img)

    bicubic_img_path = "{}/{:02d}-bicubic-psnr({:0.2f}).bmp".format(bicubic_dir, i, bicubic_psnr)
    cv2.imwrite(bicubic_img_path, bicubic_img)
    
    smooth_img_path = "{}/{:02d}-smooth-psnr({:0.2f}).bmp".format(smooth_dir, i, smooth_psnr)
    cv2.imwrite(smooth_img_path, smooth_img)

