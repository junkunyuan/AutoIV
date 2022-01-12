# AutoIV

## Introduction

This repository contains the implementation code for paper:

**Auto IV: Counterfactual Prediction via Automatic Instrumental Variable Decomposition**

Junkun Yuan, Anpeng Wu, Kun Kuang, Bo Li, Runze Wu, Fei Wu, Lanfen Lin

*ACM Transactions on Knowledge Discovery from Data (TKDD), 2022*

[[arXiv](https://arxiv.org/abs/2107.05884)]

## Brief Abstract for the Paper

<p align="center">
    <img src="framework.png" width="150"> <br>
</p>

Instrumental Variable (IV) is a powerful tool for causal inference, but it is hard to find/pre-define valid IVs. We propose an Automatic Instrumental Variable decomposition (AutoIV) algorithm to generate IV representations from observed variables through mutual information constraints for IV-based counterfactual prediction.

## Requirements

You may need to build suitable Python environment for the experiments.

The following package versions are recommened.

* python 3.6
* tensorflow-gpu 1.15.0

Device:

* GPU with VRAM > 3GB (strictly).
* Memory > 4GB.

## Usage ***

1. Configure ***run.sh*** file.
2. Run the code with command:

```
nohup sh run.sh > run.txt 2>&1 &
```

3. Your may check the results in the following path:

| Information                 | Path to check                                                                               | Note                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Generated synthetic data    | ***data/"dataset"/"dataset"-train_"data_num"/***                                            | $x$: treatment; $y$: structural response; $ye$: true outcome.                                                                                                                                                                                                                                                                                                             |
| Generated IV representation | ***data/"dataset"/autoiv-"dataset"/autoiv-"dataset"-train_"data_num"-rep_"rep_dim"/data/*** | column 1:$x$ treatment;<br />column2: $x_{pre}$ treatment predicted by treatment regression network;<br />column 3: $y$ true outcome;<br />column 4: $y_{pre}$ outcome predicted by outcome regression network;<br />columns 5~{4+"rep_dim"}: generated representation of IV $Z$;<br />columns {5+"rep_dim"}~{4+2*"rep_dim"}: generated representation of confounder $C$. |
| Training details of AutoIV  | ***AutoIV-results/"dataset"-train_"data_num"-date/***                                       | The value of each loss and MSE error of training, validation, and test,<br />during training process.                                                                                                                                                                                                                                                                     |

## Updates

- [01/09/2021] Our work is published on TKDD 2022. See https://dl.acm.org/doi/10.1145/3494568.

## Citation

If you find our code or idea useful for your research, please consider citing our work.

```bib
@article{yuan2022autoiv,
author = {Yuan, Junkun and Wu, Anpeng and Kuang, Kun and Li, Bo and Wu, Runze and Wu, Fei and Lin, Lanfen},
title = {Auto IV: Counterfactual Prediction via Automatic Instrumental Variable Decomposition},
year = {2022},
volume = {16},
number = {4},
issn = {1556-4681},
doi = {10.1145/3494568},
journal = {ACM Trans. Knowl. Discov. Data}
```

## Contact

If you have any questions, feel free to contact us through email (yuanjk@zju.edu.cn or anpwu@zju.edu.cn) or GitHub issues. Thanks!
