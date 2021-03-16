import logging
import tensorflow as tf
from data import get_dataset, get_train_pipeline
from training import train
from model_small import BIGBIGAN_G, BIGBIGAN_D_F, BIGBIGAN_D_H, BIGBIGAN_D_J, BIGBIGAN_E
import matplotlib.pyplot as plt
from misc import get_fixed_random, generate_images, _data2plot
import numpy as np
def extract_features(config):
    # Setup tensorflow
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


    # Load dataset
    logging.info('Getting dataset...')
    train_data, test_data = get_dataset(config)

    # setup input pipeline
    logging.info('Generating input pipeline...')
    train_data = get_train_pipeline(train_data, config)
    test_data = get_train_pipeline(test_data, config)

    # get model
    logging.info('Prepare model for training...')
    weight_init = tf.initializers.orthogonal()
    if config.dataset == 'mnist':
        weight_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)

    model_encoder = BIGBIGAN_E(config, weight_init)
    model_generator = BIGBIGAN_G(config, weight_init)
    print(config.encoder_path)
    model_encoder.load_weights(config.encoder_path)
    # print(config.encoder_path.replace("model_en",  'gen'))
    # model_generator.load_weights(config.encoder_path.replace("model_en",  'gen'))

    # train
    logging.info('Start extracting...')
    
    def get_feats(data):
        feats = []
        labels = []
        for image, label in data:
            latent_code_real = model_encoder(image, training=False)
            # z_fake = tf.random.truncated_normal([len(latent_code_real), config.num_cont_noise])
            # z = latent_code_real[:, :config.num_cont_noise] * tf.random.truncated_normal(
            #         [len(latent_code_real), config.num_cont_noise]) + latent_code_real[:, config.num_cont_noise:]
            # fake_images_z = _data2plot(model_generator(z, None, training=False), config)
            # fake_images_z.savefig('fake_images_z.png')
            # real_fig = _data2plot(image, config)
            # real_fig.savefig('real_fig.png')
            # fake_images = _data2plot(model_generator(z_fake, None, training=False), config)
            # fake_images.savefig('fake_images.png')
            feats.extend(latent_code_real.numpy())
            if type(label) == tuple:
                label = np.stack([label[0].numpy(), label[1].numpy()], axis=1)
            else:
                label = label.numpy()
            labels.extend(label)
        return np.array(feats), np.array(labels)

    train_feats, train_labels = get_feats(train_data)
    val_feats, val_labels = get_feats(test_data)

    np.save(config.features_path, [train_feats, train_labels, val_feats, val_labels])

    print(train_feats.shape, train_labels.shape)
    print(val_feats.shape, val_labels.shape)

    # Finished
    logging.info('extracting finished ;)')
