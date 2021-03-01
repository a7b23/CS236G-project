# CS236G-project

### To start the training run -  
```
python main.py --dataset ${dataset} --dataroot ${data_root} --save_model_dir ${model_dir} --save_image_dir ${image_dir}
```

### To start the training with positive augmentations run -  
```
python main_pos.py --dataset ${dataset} --dataroot ${data_root} --save_model_dir ${model_dir} --save_image_dir ${image_dir} --alpha 0.5
```

### To start the training with negatives run -  
```
python main_neg.py --dataset ${dataset} --dataroot ${data_root} --save_model_dir ${model_dir} --save_image_dir ${image_dir} --alpha 0.5
```

Here dataset can be cifar10, svhn or "cifar_mnist"

### To save the features, run -  


```
python save_features.py --dataset ${dataset}
```

### To run KNN classification, -  

```
python nearest_neighbour_acc_1.py
```

