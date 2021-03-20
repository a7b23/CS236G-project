# CS236G-project

### Training
Run the following commands for different models as described below:
#### Baseline BiGAN

```bash
python main.py 
  --result_path RESULT_PATH   # PAth to store the trained model and checkpoints     
  --dataset DATASET           # one of cifar10, svhn, cifar_mnist, timagenet
  --num_epochs 100
  --num_cont_noise 256

```

#### Extension of BiGAN with positive pair discrimination

```bash
python main_all.py --pos
  --result_path RESULT_PATH   # PAth to store the trained model and checkpoints     
  --dataset DATASET           # one of cifar10, svhn, cifar_mnist, timagenet
  --num_epochs 100
  --num_cont_noise 256
```

#### Extension of BiGAN with negative pair discrimination

```bash
python main_all.py --neg
  --result_path RESULT_PATH   # PAth to store the trained model and checkpoints     
  --dataset DATASET           # one of cifar10, svhn, cifar_mnist, timagenet
  --num_epochs 100
  --num_cont_noise 256
```

#### Extension of BiGAN with both positive and negative pair discrimination

```bash
python main_all.py --all
  --result_path RESULT_PATH   # PAth to store the trained model and checkpoints     
  --dataset DATASET           # one of cifar10, svhn, cifar_mnist, timagenet
  --num_epochs 100
  --num_cont_noise 256
```

### Evaluation

#### To save the features, run -
Generate and save the features of test dataset using a trained model for evaluation. 
Run the following script with the parameters as specified below:
```bash
python main.py 
    --encoder_path ENCODER_PATH     # Path to trained encoder model
    --features_path FEATURES_PATH   # Path to store the generated features
    --dataset DATASET               # one of cifar10, svhn, cifar_mnist_mnist, cifar_mnist_cifar, timagenet
    --num_cont_noise 256

```

After generating the features, run KNN or logistic classifier evaluation as specified below:
#### KNN classification:

```bash
python ../bihan_experiments/nearest_neighbour_acc_1.py 
    --dataset DATASET           # one of cifar10, svhn,cifar_mnist_mnist, cifar_mnist_cifar, timagenet
    --feat_dir FEAT_DIR         # path to root directory to save the features
```

#### Logistic Regression
```bash
python ../bihan_experiments/classify_linear.py --model_type logistic
    --dataset DATASET           # one of cifar10, svhn, cifar_mnist_mnist, cifar_mnist_cifar, timagenet
    --feat_dir FEAT_DIR         # path to root directory to save the features

```