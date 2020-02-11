# TensorFlow YOLOV3

### Pure TensorFlow 2.0 implementation of the Pix2Pix model - built of the Google tutorial
#### Features
- CPU and GPU use
- Training, inference, and evaluation usage
- Tensorflow 2.0 Best Practices
- Proper packaging

#### Installation Instructions
##### *tensorflow-gpu* v2.0.0
at this time, tensorflow-gpu v2.0.0 requires CUDA 10.0 and cuDNN 7.6.4
For arch users, follow the following links and download the proper packages. (
- [cuda](https://archive.org/download/archlinux_pkg_cuda)
  - [pk.tar.xz link](https://archive.org/download/archlinux_pkg_cuda/cuda-10.0.130-2-x86_64.pkg.tar.xz)
- [cudnn] (https://archive.org/download/archlinux_pkg_cudnn)
  - [pkg.tar.xz link](https://archive.org/download/archlinux_pkg_cudnn/cudnn-7.6.4.38-1-x86_64.pkg.tar.xz)
 Running `sudo pacman -U pkg-file-name`

 For ubunutu users, follow the following [link](https://www.tensorflow.org/install/gpu)

 **I believe CUDA 10.1 is now supported however I have not tried this**

##### package installation - from source
- git clone this repository
- pip install the `requirements.txt`
- pip install the package


#### Usage
- run `python -m TFPix2Pix train --help' for instructions
```
usage: __main__.py train [-h] --data DATA --checkpoint CHECKPOINT --log-dir
                         LOG_DIR --image-direction {AtoB,BtoA} --epochs EPOCHS
                         [--batch-size BATCH_SIZE] [--buffer-size BUFFER_SIZE]
                         [--lambda _LAMBDA] [--save-freq SAVE_FREQ]
                         [--input-shape INPUT_SHAPE] [--gpu] [--tensorboard]
                         [--eager]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Required. Dataset path
  --checkpoint CHECKPOINT
                        Required. Checkpoint path
  --log-dir LOG_DIR     Required. Log dir path
  --image-direction {AtoB,BtoA}
                        Required. Image Direction
  --epochs EPOCHS       Required. Number of epochs to train for
  --batch-size BATCH_SIZE
                        Default = 1. Batch Size for Training.
  --buffer-size BUFFER_SIZE
                        Default = 400. Buffer Size for Training
  --lambda _LAMBDA      Default = 100. Lambda value for Training
  --save-freq SAVE_FREQ
                        Default = 20. Save every X number of epochs
  --input-shape INPUT_SHAPE
                        Default = (256, 256, 3). Input Shape.
  --gpu                 Default = False. Set if using gpu
  --tensorboard         Default = False. Set if using tensorboard
  --eager               Default = False. Set if using eager execution
```
- run `python -m TFPix2Pix test --help' for instructions
```
usage: __main__.py test [-h] --weights WEIGHTS --input INPUT --output OUTPUT
                        [--batch-size BATCH_SIZE] [--gpu]
                        [--input-shape INPUT_SHAPE]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     Required. Weights file path
  --input INPUT         Required. Input images dir path
  --output OUTPUT       Required. Output images dir path
  --batch-size BATCH_SIZE
                        Default = 1. Batch Size for Testing.
  --gpu                 Default = False. Set if using gpu
  --input-shape INPUT_SHAPE
                        Default = (256, 256, 3). Input Shape.
```


#### TODO
- [x] Build the network
- [x] GPU Acceleration
- [ ] Inference demo code
- [x] Training pipeline
  - [x] fix model saving issue
  - [ ] eager mode (`tf.GradientTape`)
  - [ ] graph mode (`model.fit`)
  - [ ] tf.function() slow issue
- [x] Tensorflow Best Practices
  - subclass `tf.keras.Model` and `tf.keras.layers.Layer`
