import tensorflow as tf

import tensorflow_datasets as tfds
from data_util import preprocess_image

NUM_CALLS = tf.data.experimental.AUTOTUNE
NUM_PREFETCH = tf.data.experimental.AUTOTUNE

mnist = tf.keras.datasets.mnist
(mnist_images, mnist_labels), _ = mnist.load_data()


def map_fn(image, label):
    # Sample one mnist image.
    i = tf.random.uniform([], maxval=len(mnist_images), dtype=tf.int32)
    digit = tf.squeeze(tf.slice(mnist_images, [i, 0, 0], [1, 28, 28]))
    digit_label = tf.squeeze(tf.slice(mnist_labels, [i], [1]))
    digit = tf.image.grayscale_to_rgb(tf.expand_dims(digit, -1))
    digit = tf.image.convert_image_dtype(digit, dtype=tf.float32)
    digit = tf.image.resize(digit, [8, 8])
    image = tf.image.resize(image, [32, 32]) / 255.

    size_big, size_small = 32, 8
    images = []
    for pad_x, pad_y in [(2, 2), (2, 22), (12, 12), (22, 2), (22, 22)]:
        x_max, y_max = size_big - size_small, size_big - size_small
        d = tf.pad(digit,
                   [[pad_x, x_max - pad_x],
                    [pad_y, y_max - pad_y],
                    [0, 0]])
        images.append(d)
    images.append(image)

    image = tf.reduce_max(tf.stack(images, 0), 0)
    return image, (label, digit_label)


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, [32, 32])
    image_aug = preprocess_image(image, 32, 32, is_training=True)
    image = tf.concat([image, image_aug], axis=-1)
    
    return image, label


def get_dataset(config):
    if config.dataset != 'cifar_mnist':
        datasets, ds_info = tfds.load(name=config.dataset, with_info=True, as_supervised=True, data_dir=config.dataset_path)
    else:
        datasets, ds_info = tfds.load(name='cifar10', with_info=True, as_supervised=True,
                                      data_dir=config.dataset_path)
        for k in list(datasets.keys()):
            datasets[k] = datasets[k].map(map_fn)
    train_data, test_data = datasets['train'], datasets['test']
    return train_data, test_data


def get_train_pipeline(dataset, config):
    if config.dataset != 'cifar_mnist':
        dataset = dataset.map(scale, num_parallel_calls=NUM_CALLS)
    if (config.cache_dataset):
        dataset = dataset.cache()
    dataset = dataset.shuffle(config.data_buffer_size).batch(config.train_batch_size, drop_remainder=True).prefetch(
        NUM_PREFETCH)
    return dataset
