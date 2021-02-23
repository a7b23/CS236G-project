# CS236G-project

### To save w along with the corresponding generated images run -  
```
python save_w.py --dataset cats --total_images 50000
```

### To fit a distribution over the sampled w, run -  

For the KDE fitting -  

```
python compute_likelihood.py --model_type kde --bandwidth scott
```

For GMM fitting -  

```
python compute_likelihood.py --model_type gm --mixtures 100
```

### To select samples on the basis of likelihood scores, -  

For KDE - 
```
python select_samples.py --model_type kde --bandwidth scott --algorithm top_k
```

For GMM - 
```
python select_samples.py --model_type gm --mixture 100 --algorithm top_k
```
