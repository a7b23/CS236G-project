from __future__ import print_function
import argparse
import cv2
import lmdb
import numpy
import os
from os.path import exists, join
def export_images(db_path, out_dir, flat=False, limit=-1):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            if not flat:
                image_out_dir = join(out_dir, '/'.join(key[:6]))
            else:
                image_out_dir = out_dir
            if not exists(image_out_dir):
                os.makedirs(image_out_dir)
            image_out_path = join(image_out_dir, key.decode() + '.webp')
            with open(image_out_path, 'wb') as fp:
                fp.write(val)
            count += 1
            if count == limit:
                break
            if count % 1000 == 0:
                print('Finished', count, 'images')



if __name__ == "__main__":
    export_images('/atlas/u/a7b23/data/lsun/cat', '/atlas/u/a7b23/data/lsun/cat_data', True, limit = 50000)

