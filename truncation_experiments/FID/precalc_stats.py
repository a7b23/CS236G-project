#!/usr/bin/env python3

import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import fid
import tensorflow as tf
from PIL import Image
########
# PATHS
########
#data_path = '' # set path to training set images
#output_path = 'cifar10_train_10k_random_stats.npz' # path for where to store the statistics
#output_path = 'CA64_test_stats.npz'
#output_path = 'LSUN_bedroom_stats.npz'
output_path = 'LSUN_cat_stats.npz'
#output_path = 'CAHQ128_3k_stats.npz'
#output_path = 'CAHQ128_3k_stats.npz'
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = None
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

# loads all images into memory (this might require a lot of RAM!)
#print("load images..", end=" " , flush=True)
#data_path = "../data/lsun_bedroom/"
data_path = "../data/lsun_cat/"
#image_list = glob.glob(os.path.join(data_path, '*.jpg'))
image_list = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]
images = np.array([np.array(Image.open(fn).resize((256, 256))) for fn in image_list])
print("%d images found and loaded" % len(images))
print(images.shape, np.min(images), np.max(images))
#images = np.load("cinic10_normal.npy")
#images = np.load('CAHQ128_normal_3k.npy')

#images = np.load("cifar10_normal_50k.npy")
#images = np.load("../winter_2021/NVAE/images/CA_test_images_16_orig.npy")

np.random.shuffle(images)
images = images[:50000]


print("create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")

print("calculte FID stats..", end=" ", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")

