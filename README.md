# Domain Adaption using Graph Embedding (DAGE)
This repository supplies an implementation of a novel domain adaptation method dubbed Domain Adaptation using Graph Embedding (DAGE).

We additionally provide implementations of the following baseline transfer learning and domain adaptation methods:
* Fine tuning using gradual unfreeze
* [CCSA](https://arxiv.org/abs/1906.00684)
* [d-SNE](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_d-SNE_Domain_Adaptation_Using_Stochastic_Neighborhood_Embedding_CVPR_2019_paper.pdf)


## Datasets and abbreviation
| Experiments | Datasets |
| ----------- |:--------:|
| Office-31   | AMAZON (A), DSLR (D), and WEBCAM (W)         |

## Setup

### Install dependencies
```bash
$ pip install -r requirements.txt
```

### Download datasets
```bash
$ ./scripts/get_office.sh
```


## Running the code
```run.py``` is the entry-point for running the implemented methods.
A number of scripts are supplied, with which one can test different methods and configurations.

An example which tunes a model on source data, and tests on target data is
```bash
$ sh scripts/office31_tune_source.sh
```

A short description is included in each script.

## Hyper-parameter optimisation
A separate python project ```hypersearch.py``` can be used to perform a hyper-parameter search using Bayesian Optimisation.


## Results
A number of notebooks are supplied, in which one can visualise the Office-31 data, see the results of our experiments, and the conducted hyper-parameter search.

## Authors

* **Lukas Hedegaard** - https://github.com/lukashedegaard
* **Omar Ali Sheikh-Omar** -  https://github.com/sheikhomar

