# Building CodAdapt

A progressive educational project that rebuilds selected ideas behind **CodAdapt from scratch**, starting from a one-feature bucket-memory model and evolving step by step toward coded memory, adaptive codes, collisions and shared coded memory across multiple features.

The project uses the **Kaggle Titanic dataset** as a constant practical laboratory. Keeping the same dataset across all versions makes every architectural change easy to inspect and compare.

The implementation intentionally avoids using the CodAdapt package itself. Bucketization, memory construction, code assignment, compact addressing and prediction are implemented directly with NumPy and pandas so that the mechanics remain visible.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lucalullo/building-codadapt)

[Kaggle notebook](https://www.kaggle.com/code/lucalullo/building-codadapt) · [GitHub repository](https://github.com/lucalullo/building-codadapt) · [Technical report](project-report-en.pdf)

![Project roadmap and architecture](infographic.png)

## Overview

The project follows one main principle:

> **One version, one major concept.**

Version 1 introduces **Bucket Memory**:

1. use `Age` as the only input feature;
2. fill missing ages with the training median;
3. divide continuous values into four quantile buckets;
4. learn the average target value for each bucket;
5. store the learned values in a small memory array;
6. use the bucket index directly as the memory address;
7. convert memory values into binary Titanic predictions with a `0.5` threshold.

Version 2 adds **Coded Memory** while preserving the rest of the model:

## Current capabilities

- load the Titanic training and test datasets;
- use numerical features directly with NumPy and pandas;
- fill numerical missing values with medians learned only from the training set;
- discretize continuous features with quantile-based bucket boundaries;
- reuse the same learned medians and bucket boundaries at prediction time;
- compute target means for individual buckets;
- assign simple data-dependent binary codes to buckets;
- separate bucket identity from memory address;
- compress several bucket states into fewer memory cells;
- create intentional collisions between discrete states;
- combine codes from multiple features into a shared memory address;
- learn one target value per shared-memory cell;

## Architecture

```text
Titanic Data
    ↓
Training Medians
Age / Fare
    ↓
Quantile Bucketization
4 buckets per feature
    ↓
Bucket Target Means
    ↓
Simple Adaptive Codes
0 or 1 per bucket
    ↓
Feature Codes
Age code + Fare code
    ↓
Shared Address
age_code + 2 × fare_code
    ↓
Shared Memory
4 cells
    ↓
Memory Lookup
    ↓
Continuous Prediction
    ↓
Threshold 0.5
    ↓
Binary Titanic Prediction
    ↓
Kaggle Submission
```

The key transition in Version 4 is that multiple independently discretized features contribute codes to the same shared memory address. This makes it possible to represent a compact interaction without allocating one separate memory cell for every raw pair of feature buckets.

## Version guide

| Version | Focus | Notebook and resources |
|---:|---|---|
| 1 | Bucket Memory | [`v1-bucket-memory/`](v1-bucket-memory/) |
| 2 | Coded Memory | [`v2-coded-memory/`](v2-coded-memory/) |
| 3 | Adaptive Codes | [`v3-adaptive-codes/`](v3-adaptive-codes/) |
| 4 | Shared Coded Memory | [`v4-shared-coded-memory/`](v4-shared-coded-memory/) |

### What each version introduces

<details>
<summary>Version 1 - Bucket Memory</summary>

Version 1 establishes the smallest complete learning loop.

`Age` is the only input feature. Missing values are filled with the median learned from the training set, then the feature is divided into four quantile buckets.

</details>

<details>
<summary>Version 2 - Coded Memory</summary>

Version 2 keeps the same bucketization and memory values but separates the bucket from the memory address.

The initial mapping is deliberately simple:

</details>

<details>
<summary>Version 3 - Adaptive Codes</summary>

Version 3 makes the codes data-dependent and reduces the memory from four cells to two.

For every bucket, the model computes the local target mean and compares it with the global target mean. The bucket receives code `1` when its mean is greater than or equal to the global mean, otherwise code `0`.

</details>

<details>
<summary>Version 4 - Shared Coded Memory</summary>

Version 4 extends the model to two numerical features:

Each feature is independently bucketized and assigned simple binary codes. The two feature codes are then combined into one shared address:

</details>

## Observed behavior

This project is educational rather than leaderboard-oriented. The notebooks generate valid Titanic submission files, but the repository does not currently use Kaggle leaderboard score as the main measure of progress.

The most useful progression is architectural:

| Version | Memory structure | Addressing |
|---|---|---|
| V1 | 4 cells | bucket index |
| V2 | 4 cells | fixed code |
| V3 | 2 cells | learned binary code with collisions |
| V4 | 4 shared cells | combined codes from Age and Fare |

This makes the change in representation visible even when classification quality is intentionally secondary.

## Related project

The educational notebooks isolate only a small subset of the ideas used by the real library.

The production CodAdapt estimator is substantially more sophisticated. Its documented native representation includes concepts such as:

- an intercept;
- univariate bucket effects;
- coded interaction tables;
- integer feature codes combined through compact addresses;
- modular addressing;
- multiple interaction tables;
- adaptive code optimization;
- iterative score or residual updates;
- validation and early stopping;
- numerical and categorical preprocessing;
- shared multi-resolution encoding.

Those mechanisms are intentionally not reproduced all at once here. The purpose of **Building CodAdapt** is to expose one understandable mechanism at a time.

## Getting started

### Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Titanic `train.csv` and `test.csv` data

The notebooks were developed for the Kaggle Titanic environment and currently read the competition files from:

```python
PATH = "/kaggle/input/competitions/titanic/"
```

To run locally, install the required libraries:

```bash
pip install numpy pandas jupyter
```

Clone the repository:

```bash
git clone https://github.com/lucalullo/building-codadapt.git
cd building-codadapt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Download the Titanic `train.csv` and `test.csv` files, then update the `PATH` variable in the notebook to point to their local directory.

Open the version you want to study and run the cells in order.

For example, to study the current shared coded-memory implementation:

```text
v4-shared-coded-memory/building-codadapt.ipynb
```

The Titanic dataset itself is not included in this repository.

## Repository structure

Each version is kept in its own folder with the corresponding notebook and, where available, version-specific diagrams and reports.

## Documentation

- [Project report (English)](project-report-en.pdf)
- [Relazione del progetto (Italiano)](project-report-it.pdf)
- [Project infographic](infographic.png)

## License

Distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
