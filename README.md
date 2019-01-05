# README

General usage:

    python comparisons.py run_denmo $dset $dilations $erosions $epochs
    python comparisions.py run_baseline $model $dset $width $epochs
    
Run `scripts/training2csv.py` to convert Tensorflow logs to result CSVs.

## Getting tensorboard working
* `pip install tensorboard`
* `tensorboard --logdir training-logs/`


### Table 2: MNIST Accuracy / Figure 4
```
python comparisons.py run_denmo mnist --epochs=400 --dilations=5 --erosions=5
python comparisons.py run_denmo mnist --epochs=400 --dilations=25 --erosions=25
python comparisons.py run_denmo mnist --epochs=400 --dilations=50 --erosions=50
python comparisons.py run_denmo mnist --epochs=400 --dilations=100 --erosions=100
python comparisons.py run_denmo mnist --epochs=400 --dilations=400 --erosions=0
python comparisons.py run_denmo mnist --epochs=400 --dilations=0 --erosions=400
```

### Table 3:
```
python comparisons.py run_denmo mnist --epochs=300 --dilations=200 --erosions=200
python comparisons.py run_denmo fashion_mnist --epochs=300 --dilations=400 --erosions=400
```

### Table 4 / Figure 5
```
python comparisons.py run_denmo cifar10 --epochs=150 --dilations=100 --erosions=100
python comparisons.py run_baseline tanh cifar10 --epochs=150 --h-layers=200
python comparisons.py run_baseline relu cifar10 --epochs=150 --h-layers=200
python comparisons.py run_baseline maxout cifar10 --epochs=150 --h-layers=200

python comparisons.py run_denmo cifar10 --epochs=150 --dilations=200 --erosions=200
python comparisons.py run_baseline tanh cifar10 --epochs=150 --h-layers=400
python comparisons.py run_baseline relu cifar10 --epochs=150 --h-layers=400
python comparisons.py run_baseline maxout cifar10 --epochs=150 --h-layers=400
```

#### Note: this is used in Figure 5
```
python comparisons.py run_denmo cifar10 --epochs=150 --dilations=300 --erosions=300
python comparisons.py run_denmo cifar10 --epochs=150 --dilations=0 --erosions=600
python comparisons.py run_denmo cifar10 --epochs=150 --dilations=600 --erosions=0
python comparisons.py run_baseline tanh cifar10 --epochs=150 --h-layers=600
python comparisons.py run_baseline relu cifar10 --epochs=150 --h-layers=600
python comparisons.py run_baseline maxout cifar10 --epochs=150 --h-layers=600
```

### Table 1: Circle
#### Omitted
