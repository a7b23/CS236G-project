# CS236G-project

### Training
Run the following commands for different models as described below:
#### Baseline BiGAN

##### Without Extra Image Discriminator
```bash
python main.py --use_image_discriminator False --batch_size 100 --num_epochs 100  --use_cuda --cuda_device 0
  --dataset DATASET                 # one of cifar10, svhn, cifar_mnist, timagenet
  --dataroot DATAROOT               # root path of the datasets
  --save_model_dir SAVE_MODEL_DIR   # path to directory to save model checkpoints
  --save_image_dir SAVE_IMAGE_DIR   # path to directory to save generated images
```

##### With Extra Image Discriminator
```bash
python main.py --use_image_discriminator True --batch_size 100 --num_epochs 100  --use_cuda --cuda_device 0
  --dataset DATASET                 # one of cifar10, svhn, cifar_mnist, timagenet
  --dataroot DATAROOT               # root path of the datasets
  --save_model_dir SAVE_MODEL_DIR   # path to directory to save model checkpoints
  --save_image_dir SAVE_IMAGE_DIR   # path to directory to save generated images
```

#### Extension of BiGAN with positive pair discrimination

```bash
python main_pos.py --use_cuda
  --dataset DATASET                 # one of cifar10, svhn, cifar_mnist, timagenet
  --alpha ALPHA                     # Weightage of the extra reals` loss
  --dataroot DATAROOT               # root path of the datasets
  --save_model_dir SAVE_MODEL_DIR   # path to directory to save model checkpoints
  --save_image_dir SAVE_IMAGE_DIR   # path to directory to save generated images
```

#### Extension of BiGAN with negative pair discrimination

```bash
python main_neg.py --use_cuda
  --dataset DATASET                 # one of cifar10, svhn, cifar_mnist, timagenet
  --alpha ALPHA                     # Weightage of the extra fakes` loss
  --dataroot DATAROOT               # root path of the datasets
  --save_model_dir SAVE_MODEL_DIR   # path to directory to save model checkpoints
  --save_image_dir SAVE_IMAGE_DIR   # path to directory to save generated images
```


#### Extension of BiGAN with both positive and negative pair discrimination

```bash
python main_combined.py --batch_size 100 --num_epochs 100 --use_cuda --cuda_device 0
  --dataset DATASET                 # one of cifar10, svhn, cifar_mnist, timagenet
  --alpha ALPHA                     # Weightage of the extra reals` loss
  --beta BETA                       # Weightage of the extra fakes` loss
  --dataroot DATAROOT               # root path of the datasets
  --save_model_dir SAVE_MODEL_DIR   # path to directory to save model checkpoints
  --save_image_dir SAVE_IMAGE_DIR   # path to directory to save generated images
```

### Evaluation

#### To save the features, run -
Generate and save the features of test dataset using a trained model for evaluation. 
Run the following script with the parameters as specified below:
```bash
python save_features.py 
    --dataset DATASET           # one of cifar10, svhn, cifar_mnist_mnist, cifar_mnist_cifar, timagenet
    --feat_dir FEAT_DIR         # path to root directory to save the features
    --model_path MODEL_PATH     # path of the checkpoint for the trained model
    --dataroot DATAROOT         # root path of the datasets
```

After generating the features, run KNN or logistic classifier evaluation as specified below:
#### KNN classification:

```bash
python nearest_neighbour_acc_1.py 
    --dataset DATASET           # one of cifar10, svhn,cifar_mnist_mnist, cifar_mnist_cifar, timagenet
    --feat_dir FEAT_DIR         # path to root directory to save the features
```

#### Logistic Regression
```bash
python classify_linear.py --model_type logistic
    --dataset DATASET           # one of cifar10, svhn, cifar_mnist_mnist, cifar_mnist_cifar, timagenet
    --feat_dir FEAT_DIR         # path to root directory to save the features

```

### Visualization

To visualize the representation learned by the models, 