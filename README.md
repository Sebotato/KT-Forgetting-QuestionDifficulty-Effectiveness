# Knowledge Fading Analysis
This repository accompanies the paper "Analysis of Knowledge Fading" submitted for double-blind review at LAK25.

## Overview

### Key Features


## Quick Start
### Installation
All implementations are conducted via the publicly available [EduStudio](https://edustudio.ai) library. EduStudio is compatible with the following operating systems: Linux, Windows 10 and macOS X. 

Python 3.8 (or later) and torch 1.10.0 (or later) are required to install EduStudio. There are two ways to install EduStudio.

```bash
pip install -U edustudio
```
### Data Pre-processing
Data is pre-processed via a sequence of atomic operations. The first atomic operation, inheriting the protocol class `BaseRaw2Mid`, processes raw data into middle data. The following atomic operations, inheriting the protocol class `BaseMid2Cache`, construct the process from middle data to cache data. Cache data is in the format convenient for model usage. For demonstration purposes we have pre-processed data from the `BA06` and `NIPS34` datasets into middle data. For all other datasets, the data can be downloaded from their original sources provided in the References section and pre-processed using the [Raw2Mid](https://edustudio.readthedocs.io/en/latest/features/atomic_operations.html) data atomic operations as below:

```bash
edustudio r2m R2M_#DatasetName --dt #DatasetName --rawpath data/#DatasetName/rawdata --midpath data/#DatasetName/middata
```

### Run DKT Models
To execute a decay function, adjust the `decay_function` configuration in the Python file. The table below lists all available decay configuration options. For AKT, HawkesKT, RKT, and CT_NCM, the default setting is `'exp'` as explained in their original papers. For DKTForget and SimpleKT, the original model does not consider a memory decay function, hence its default setting is `'non'`. 

| **Name** | **Decay configuration options** |
|----------|--------------|
| Remove forgetting | `'rem'` |
| No decay | `'non'` |
| Exponential decay | `'exp'` |
| Logarithmic decay | `'log'` |
| Sigmoid decay | `'sig'` |
| Inverse decay | `'inv'` |

Below is an example of running `AKT` with exponential decay using Python file.

Create a Python file (e.g., run.py) to run, an example as below:

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='Bridge2Algebra_0607',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
        'decay_function': 'exp'
    },
    datatpl_cfg_dict={
        'cls': 'KTInterDataTPLCptUnfold'
    },
    modeltpl_cfg_dict={
        'cls': 'AKT',
    },
    evaltpl_cfg_dict={
        'clses': 'PredictionEvalTPL',
    }
)

```

Then execute the file:

```bash
python run.py
```

DKT models' configuration settings other than decay function can be found at : https://edustudio.readthedocs.io/en/latest/user_guide/reference_table.html#kt-models
## References 
- The models that have been involved in this research

| **Model Name** | **Publication Title** | **Conference** | **Year** |
|----------------|-----------------------|------------------------|----------|
| [AKT](https://dl.acm.org/doi/abs/10.1145/3394486.3403282) | Context-Aware Attentive Knowledge Tracing | KDD | 2020 |
| [HawkesKT](https://dl.acm.org/doi/10.1145/3437963.3441802) | Temporal cross-effects in knowledge tracing | WSDM | 2021 |
| [RKT](https://arxiv.org/pdf/2008.12736) | RKT: Relation-Aware Self-Attention for Knowledge Tracing | CIKM | 2020 |
| [CT-NCM](https://www.semanticscholar.org/paper/Reconciling-Cognitive-Modeling-with-Knowledge-A-Ma-Wang/d3b4115906be4939b7f93736090ec1844d9ae591) | Reconciling Cognitive Modeling with Knowledge Forgetting: A Continuous Time-aware Neural Network Approach | IJCAI | 2022 |
| [DKTForget](https://dl.acm.org/doi/10.1145/3308558.3313565) | Augmenting Knowledge Tracing by Considering Forgetting Behavior | WWW | 2019 |
| [SimpleKT](https://arxiv.org/abs/2302.06881) | simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing | ICLR | 2023 |

- A summary of information for the selected datasets

| **Dataset** | **Questions** | **Knowledge Concepts** | **Students** | **Answers** | **Timestamp** | **Purpose** | **Questions Bundling** |
|-------------|---------------|------------------------|--------------|-------------|---------------|-------------|------------------------|
| [ASSIST12](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) | 45,716 | 265 | 27,066 | 2,541,201 | ✓ | Educational research | ✓ |
| [ASSIST17](https://sites.google.com/view/assistmentsdatamining/dataset?authuser=0) | 3,162 | 102 | 1,709 | 942,816 | ✓ | Educational longitudinal prediction | ✓ |
| [BA06](https://pslcdatashop.web.cmu.edu/KDDCup/) | 207,856 | 493 | 1,146 | 3,656,871 | ✓ | Mathematics-focused educational prediction |  |
| [NIPS34](https://eedi.com/projects/neurips-education-challenge) | 948 | 57 | 4,918 | 1,382,727 | ✓ | Educational research on diagnostic questions |  |
| [EN1](https://github.com/riiid/ednet) | 13,169 | 188 | 784,309 | 95,293,926 | ✓ | AI tutoring | ✓ |
