# Domain Adaption using Graph Embedding (DAGE)
This repository supplies an implementation of a the supervised domain adaptation method [Domain Adaptation using Graph Embedding (DAGE)]([paper](https://arxiv.org/abs/2003.04063).).

We additionally provide implementations of the following baseline transfer learning and domain adaptation methods:
* Fine tuning using gradual unfreeze
* [CCSA](https://arxiv.org/abs/1906.00684)
* [d-SNE](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_d-SNE_Domain_Adaptation_Using_Stochastic_Neighborhood_Embedding_CVPR_2019_paper.pdf)


## Datasets and abbreviation
| Experiments | Datasets |
| ----------- |:--------:|
| Office-31   | AMAZON (A), DSLR (D), and WEBCAM (W)         |
| MNIST -> USPS   | MNIST (M), USPS (U)         |

## Setup

### Install dependencies
```bash
$ conda env create --file environment.yml
$ conda activate dage
```

### Download datasets
```bash
$ ./scripts/get_office.sh
$ ./scripts/get_digits.sh
```


## Running the code
```run.py``` is the entry-point for running the implemented methods. 
To retreive a list of valid arguments, use ```python run.py --help```.

A number of ready-to-run scripts are supplied (found in the `scripts` folder), with which one can test different methods and configurations.

An example which tunes a model on source data, and tests on target data is
```bash
$ ./scripts/office31_tune_source.sh
```
Running DAGE of Office31 with tuned hyperparameters is acheived by using.
```bash
$ /scripts/office31_dage_lda_tuned_vgg16.sh
```



## Hyper-parameter optimisation
A separate python entry-point ```hypersearch.py``` can be used to perform a hyper-parameter search using Bayesian Optimisation.


## Results
A number of notebooks are supplied, in which one can visualise the Office-31 data, see the results of our experiments, and the conducted hyper-parameter search.

## Authors

* **Lukas Hedegaard** - https://github.com/lukashedegaard
* **Omar Ali Sheikh-Omar** -  https://github.com/sheikhomar

