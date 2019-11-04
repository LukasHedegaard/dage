# Domain Adaption
An implementation of the domain adaption methods
* Fine tuning
* [CCSA](https://arxiv.org/abs/1906.00684)
* [d-SNE](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_d-SNE_Domain_Adaptation_Using_Stochastic_Neighborhood_Embedding_CVPR_2019_paper.pdf)

## Dependencies:
```bash
$ conda env create -f environment.yml
$ conda activate da
```

## Datasets and abbreviation
| Experiments | Datasets |
| ----------- |:--------:|
| Office-31   | AMAZON (A), DSLR (D), and WEBCAM (W)         |

<!-- | Digits      | MNIST(MT), MNISTM(MM), SVHN(SN), and USPS(US)| -->
<!-- | VisDA       | Synthetic (S) and Real (R)                   | -->

In relation to domain adaption, the terminology of source (S) and target (T) is used.

## Setup

### Download datasets
```bash
$ scripts/download_office_datasets.sh
```

<!-- ### Download pretrained models
```bash
$ scripts/download_pretrained_models.sh
``` -->

## Usage
### Predifined runs
Tune a pretrained model on source data, test on target data
```bash
$ sh scripts/office31_tune_source.sh
```

Tune a pretrained model on source and target data, test on target data
```bash
$ sh scripts/office31_tune_both.sh
```

Perform domain adaption using the CCSA method
```bash
$ sh scripts/office31_ccsa.sh
```

Perform domain adaption using the d-SNE method
```bash
$ sh scripts/office31_dsne.sh
```

<!-- ### Custom runs
TODO: desccribe parameters list -->

<!-- ## Deployment -->
<!-- ## Additional setup notes -->
<!-- ## Troubleshooting -->
<!-- ## Built With -->
<!-- ## Versioning -->


## Authors

* **Lukas Hedegaard** - https://github.com/lukashedegaard
* **Omar Ali Sheikh-Omar** -  https://github.com/sheikhomar

<!-- ## License -->

<!-- ## Acknowledgments -->
