# [TensorFlow] Super-Resolution CNN 

Implementation of SRCNN model in **Image Super-Resolution using Deep Convolutional Network** paper by Tensorflow, Keras. We used Adam with optimize tuned hyperparameters instead of SGD + Momentum.


## Contents
- [Requirements](#requirements)
- [Train](#train)
- [Test](#test)
- [Demo](#demo)
- [References](#references)



## Requirements
- Python 3.8 or 3.9 
- Tensorflow 2.5.0
- Numpy 1.19.1  
- Matplotlib 3.4.3
- Pandas 1.3.4
- OpenCV 4.5.3



## Train
You **MUST** generate data first:
```
python GenerateData.py --dataset_path="dataset/" --input_size=33 --chanels=3
```
- **dataset_path**: path to dataset directory.
- **input_size**: size of input subimages (does not affect to input size of model).
- **chanels**: Number of color chanels.

After generating data, you can run this command to begin the training:
```
python Train.py --epoch=2500 --batch_size=128
```

**NOTE**: if you want to train a new model, you can delete all files in **checkpoint** directory. Your checkpoint will be saved when above command finishs and can be used for next times, so you can train this model on Colab without taking care of GPU limit.



## Test
After Training, you can test the model with this command and see the results in **test** directory:
```
python Test.py
```

We trained 60000 epochs and evaluated model with Set5 and Set14 dataset by PSNR:

<div align="center">

| Methods               | Set5 x2 | Set5 x3 | Set5 x4 | Set14 x2 | Set14 x3 | Set14 x4 |
|:---------------------:|:-------:|:-------:|:-------:|:--------:|:--------:|:--------:|
| Bicubic Interpolation | 35.7642 |	33.4815	| 31.5064 |	32.7529	 | 30.4050  | 28.8816  |
| Resize + Smoothing	| 35.3901 |	33.0289	| 31.2894 |	31.9534  | 30.0626  | 28.7333  |
| SRCNN                 | 36.7126 | 34.5102 | 32.0289 |	33.0441	 | 31.0303	| 29.2228  |

</div>


## Demo
You can put your images to **demo/images** directory first and run this command to see the results in **demo/results** directory:
```
python Demo.py --scale=2
```


We tested with some images from following links and got some results:
- https://www.pixiv.net/en/artworks/67083950
- https://mocah.org/306004-anime-girl-pink-hair-yae-sakura-honkai-impact-3rd-4k.html
- https://images.app.goo.gl/eB7PazUsSfYamCES7

<div align="center">
  <img src="./Images for README/butter.png" width="1000">  
  <p><strong>Bicubic x3 (left), resize+smoothing x3 (center), SRCNN x3 (right).</strong></p>
</div>

<div align="center">
  <img src="./Images for README/girl.png" width="1000">  
  <p><strong>Bicubic x3 (left), resize+smoothing x3 (center), SRCNN x3 (right).</strong></p>
</div>

<div align="center">
  <img src="./Images for README/sakura.png" width="1000">  
  <p><strong>Bicubic x3 (left), resize+smoothing x3 (center), SRCNN x3 (right).</strong></p>
</div>




## References
- Image Super-Resolution Using Deep Convolutional Networks: https://arxiv.org/pdf/1501.00092.pdf
- SRCNN Matlab code: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
- T91 and BSDS100 dataset: http://vllab.ucmerced.edu/wlai24/LapSRN/
- Set5 and Set14 dataset: https://github.com/jbhuang0604/SelfExSR
- YeongHyeon/Super-Resolution_CNN: https://github.com/YeongHyeon/Super-Resolution_CNN
- aditya9211/Super-Resolution-CNN: https://github.com/aditya9211/Super-Resolution-CNN
