# EfficientNAS
Code for the paper
> [Towards Automated Deep Learning: Efficient Joint Neural Architecture and Hyperparameter Search](https://arxiv.org/abs/1807.06906)\
> Arber Zela, Aaron Klein, Stefan Falkner and Frank Hutter.\
> _arXiv:1807.06906_.

This is a follow-up work of [BOHB: Robust and Efficient Hyperparameter Optimization at Scale](http://proceedings.mlr.press/v80/falkner18a.html). We use BOHB to conduct an analysis over a joint neural architecture and hyperparameter space and demostrate the weak correlation accross training budgets far from each other. Nevertheless, our search method surprisingly finds a configuration able to achieve 3.18% test error in just 3h of training.

## Requirements
```
Python >= 3.6.x, PyTorch == 0.3.1, torchvision == 0.2.0, hpbandster, ConfigSpace
```

## Running the joint search
The code is only compatible with CIFAR-10, which will be automatically downloaded, however it can be easily extended to other image datasets with the same resolution, such as CIFAR-100, SVHN, etc.

For starting BOHB one has to specify 5 parameters: `min_budget`, `max_budget`, `\eta`, `num_iterations` and `num_workers`. You can change them in the script `BOHB-CIFAR10.sh`.
NOTE: We used the [Slurm Workload Manager](https://slurm.schedmd.com/) environment to run our jobs, but it can be easily adapted to other job scheduling systems.

To start the search with the default settings (`min_budget=400`, `max_budget=10800`, `\eta =3`, `num_iterations=32`, `num_workers=10`) used in the paper just run:

```
sbatch BOHB-CIFAR10.sh
```

## Citation
```
@inproceedings{zela-automl18,
  author    = {Arber Zela and
               Aaron Klein and
               Stefan Falkner and 
               Frank Hutter},
  title     = {Towards Automated Deep Learning: Efficient Joint Neural Architecture and Hyperparameter Search},
  booktitle = {ICML 2018 AutoML Workshop},
  year      = {2018},
  month     = jul,
}
