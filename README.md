# Datasets used here

# CIFAR
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
							
[Download Link](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

# MNIST 
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

[Download Link](http://yann.lecun.com/exdb/mnist/)
ALTERNATIVE: The dataset is also provided by Tensorflow library. Use the import statement
'''
from tensorflow.keras.datasets import mnist
'''

#Flowers Dataset
This dataset contains 4242 images of flowers based on the data from flickr, google images, yandex images.
You can use this datastet to recognize plants from the photo.
The pictures are divided into five classes: chamomile, tulip, rose, sunflower, dandelion.

[Download Link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1ECTVN)

# Deep Learning Models

# Autoencoders
[Autoencoder on MNIST Dataset](https://github.com/akhil9650/Deep-Learning/blob/main/Autoencoders.ipynb)
[Autoencoder on Flower Dataset](https://github.com/akhil9650/Deep-Learning/blob/main/autoencoder_flowers.ipynb)
[Autoencoder on CIFAR Dataset](https://github.com/akhil9650/Deep-Learning/blob/main/cifar_autoencoder.ipynb)

Autoencoders compress the input into a lower-dimensional code and then reconstruct the output from this representation. The code is a compact “summary” or “compression” of the input, also called the latent-space representation.

Different model architectures ith variations in layers are tried out until we get a relatively noise-free output.

# CNN
[CIFAR CNN]](https://github.com/akhil9650/Deep-Learning/blob/main/cifar_cnn.ipynb)
[MNIST CNN](https://github.com/akhil9650/Deep-Learning/blob/main/mnist_cnn.ipynb)
Training a 4 layer CNN with dropout and batch normalization on the CIFAR dataset for image classification. 
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.

[MNIST Callbacks](https://github.com/akhil9650/Deep-Learning/blob/main/mnist_cnn_callbacks.ipynb)
This notebook implements various callbacks like early stopping and reduce learning rate on plateau, various techniques that can help improve the performance of CNN.

# GAN
[Flower GAN](https://github.com/akhil9650/Deep-Learning/blob/main/flower_gan.ipynb)
Generationg images with a 2 layer GAN with Dropout and Leaky Regularisation. 
Given a training set, this technique learns to generate new data with the same statistics as the training set. For example, a GAN trained on photographs can generate new photographs that look at least superficially authentic to human observers, having many realistic characteristics. 

