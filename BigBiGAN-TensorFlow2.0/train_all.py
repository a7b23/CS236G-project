import logging
import tensorflow as tf
from data_all import get_dataset, get_train_pipeline
from training_all import train
from model_small import BIGBIGAN_G, BIGBIGAN_D_F, BIGBIGAN_D_H, BIGBIGAN_D_J, BIGBIGAN_E
import numpy as np
import os
from PIL import Image

def save_image(img, fname):
    img = img*255.0
    img = Image.fromarray(img.astype(np.uint8))
    img.save(fname)
    
def visualize(train_data):
    out_dir = "images_pos_vis"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    for image, label in train_data:
        img, img_aug = tf.split(image, 2, axis=-1)
        images = img.numpy()
        images_aug = img_aug.numpy()
        print(images.shape, images_aug.shape, np.min(images), np.max(images), np.min(images_aug), np.max(images_aug))
        for idx, (img, img_aug) in enumerate(zip(images, images_aug)):
            if idx == 10:
                break
            save_image(img, os.path.join(out_dir, "img_" + str(idx)+".png"))
            save_image(img_aug, os.path.join(out_dir, "img_aug_" + str(idx)+".png"))
        break
        
def set_up_train(config):
    # Setup tensorflow
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


    # Load dataset
    logging.info('Getting dataset...')
    train_data, _ = get_dataset(config)

    # setup input pipeline
    logging.info('Generating input pipeline...')
    train_data = get_train_pipeline(train_data, config)
    
#     visualize(train_data)
    
    # get model
    logging.info('Prepare model for training...')
    weight_init = tf.initializers.orthogonal()
    if config.dataset == 'mnist':
        weight_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
    model_generator = BIGBIGAN_G(config, weight_init)
    model_discriminator_f = BIGBIGAN_D_F(config, weight_init)
    model_discriminator_h = BIGBIGAN_D_H(config, weight_init)
    model_discriminator_j = BIGBIGAN_D_J(config, weight_init)
    model_encoder = BIGBIGAN_E(config, weight_init)

    # train
    logging.info('Start training...')

    train(config=config,
          gen=model_generator,
          disc_f=model_discriminator_f,
          disc_h=model_discriminator_h,
          disc_j=model_discriminator_j,
          model_en=model_encoder,
          train_data=train_data)
    # Finished
    logging.info('Training finished ;)')
