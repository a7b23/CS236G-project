# CS236G-project

### Training

#### To start the training run -

```bash
python main.py --dataset ${dataset} --dataroot ${data_root} --save_model_dir ${model_dir} --save_image_dir ${image_dir}
```

#### To start the training with positive augmentations run -

```bash
python main_pos.py --dataset ${dataset} --dataroot ${data_root} --save_model_dir ${model_dir} --save_image_dir ${image_dir} --alpha 0.5
```

#### To start the training with negatives run -

```bash
python main_neg.py --dataset ${dataset} --dataroot ${data_root} --save_model_dir ${model_dir} --save_image_dir ${image_dir} --alpha 0.5
```

Here dataset can be cifar10, svhn or "cifar_mnist"

### Evaluation


#### To save the features, run -
Generate and save the features of test dataset using a trained model for evaluation. 
Run the following script with the parameters as specified below:
```bash
python save_features.py 
    --dataset DATASET           # one of cifar10,svhn,cifar_mnist_mnist,cifar_mnist_cifar,timagenet
    --feat_dir FEAT_DIR         # path to root directory to save the features
    --model_path MODEL_PATH     # path of the checkpoint for the trained model
    --dataroot DATAROOT         # root path of the datasets
```

After generating the features, run KNN or logistic classifier evaluation as specified below:
#### KNN classification:

```bash
python nearest_neighbour_acc_1.py 
    --dataset DATASET           # one of cifar10,svhn,cifar_mnist_mnist,cifar_mnist_cifar,timagenet
    --feat_dir FEAT_DIR         # path to root directory to save the features
```

#### Logistic Regression
```bash
python classify_linear.py --model_type logistic
    --dataset DATASET           # one of cifar10,svhn,cifar_mnist_mnist,cifar_mnist_cifar,timagenet
    --feat_dir FEAT_DIR         # path to root directory to save the features

```

