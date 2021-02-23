import pickle
import dnnlib
from dnnlib import tflib
import numpy as np
from PIL import Image

tflib.init_tf()

def save_image(img, fname) :
    img = Image.fromarray(img.astype(np.uint8))
    img.save(fname)

# fname = "bedroom_model/karras2019stylegan-bedrooms-256x256.pkl"
fname = "cats_model/karras2019stylegan-cats-256x256.pkl"
with open(fname, "rb") as f :
    _G, _D, Gs = pickle.load(f)

rnd = np.random.RandomState(10)
batch_size = 50
total_images = 50000

iterations = int(total_images/batch_size)

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
latent_arr = []
images_all = []
for i in range(iterations) :
    print(i, iterations)
    latents = rnd.randn(batch_size, Gs.input_shape[1])
    src_dlatents = Gs.components.mapping.run(latents, None)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.components.synthesis.run(src_dlatents, truncation_psi=1.0, randomize_noise=True, output_transform=fmt)


    # print(src_dlatents.shape)
    latent_arr.extend(src_dlatents[:,0,:])
    images_all.extend(images)

latent_arr = np.array(latent_arr)
images_all = np.array(images_all)

print(latent_arr.shape)
print(images_all.shape)
# images_all = np.array(images_all)
# print(images_all.shape, np.min(images_all), np.max(images_all))
np.save("feats_w/cats.npy", latent_arr)
np.save("images_cats/images_50000.npy", images_all)

# save_image(images[0], "images_bedroom/img0.png")
# save_image(images[1], "images_bedroom/img1.png")
