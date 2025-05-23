# Forgetting and Question Difficulty Ablation Study
This repository accompanies the paper "Effectiveness of Forgetting and Question Difficulty in Deep Knowledge Tracing" submitted at AIED 2025.

## Overview
This study conducts an ablation analysis on existing implementations of forgetting and question difficulty in DKT models by isolating these components and analysing their impact on model performance when removed. Our findings indicate that the effectiveness of forgetting and question difficulty is highly dependent on the model. Notably, modelling forgetting as an exponential decay term within self-attention mechanisms has proved to be the most effective. Additionally, incorporating question difficulty through the Rasch model into the input embeddings and integrating it into self-attention mechanisms optimises its effect. Furthermore, datasets with long-period student interaction information better capture the effects of forgetting and question difficulty, aligning with recent theories on human forgetting behaviour and the influence of question difficulty on learning.

## Quick Start
### Installation
All implementations are conducted via the publicly available [EduStudio](https://edustudio.ai) library. EduStudio is compatible with the following operating systems: Linux, Windows 10 and macOS X. 

Python 3.8 (or later) and torch 1.10.0 (or later) are required to install EduStudio.

```bash
pip install -U edustudio
```
### Data Pre-processing
Data is pre-processed via a sequence of atomic operations. The first atomic operation, inheriting the protocol class `BaseRaw2Mid`, processes raw data into middle data. The following atomic operations, inheriting the protocol class `BaseMid2Cache`, construct the process from middle data to cache data. Cache data is in the format convenient for model usage. For demonstration purposes we have pre-processed data from the `BA06` and `NIPS34` datasets into middle data. For all other datasets, the data can be downloaded from their original sources provided in the References section and pre-processed using the [Raw2Mid](https://edustudio.readthedocs.io/en/latest/features/atomic_operations.html) data atomic operations as below:

```bash
edustudio r2m R2M_#DatasetName --dt #DatasetName --rawpath data/#DatasetName/rawdata --midpath data/#DatasetName/middata
```

### Run DKT Models
To run a particular model, a Python file containing the model name, dataset name and configuration information can be constructed for execution. The configuration information should follow the guideline on [Reference Table](https://edustudio.readthedocs.io/en/latest/user_guide/reference_table.html). For models with forgetting incorporated, forgetting removal can be achieved by configuring the `forgetting` parameter in the Python file to `False`. Likewise, for models with question difficulty incorporated, question difficulty removal can be achieved by configuring the `quesDiff` parameter in the Python file to `False`.

Below is an example of running `AKT` with forgetting removed using the Python file.

Create a Python file (e.g., run.py) to run, an example as below:

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='Bridge2Algebra_0607',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
        'forgetting': False,
        'quesDiff': False
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
| [DKT+Forget](https://dl.acm.org/doi/10.1145/3308558.3313565) | Augmenting Knowledge Tracing by Considering Forgetting Behavior | WWW | 2019 |
| [simpleKT](https://arxiv.org/abs/2302.06881) | simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing | ICLR | 2023 |
| [SAINT+](https://dl.acm.org/doi/10.1145/3448139.3448188) | SAINT+: Integrating Temporal Features for EdNet Correctness Prediction | LAK | 2021 |
| [DKVMN](https://arxiv.org/abs/1611.08108) | Dynamic Key-Value Memory Networks for Knowledge Tracing | WWW | 2017 |
| [Deep-IRT](https://arxiv.org/abs/1904.11738) | Deep-IRT: Make Deep Learning Based Knowledge Tracing Explainable Using Item Response Theory | EDM | 2019 |
| [DIMKT](https://dl.acm.org/doi/abs/10.1145/3477495.3531939) | Assessing Student's Dynamic Knowledge State by Exploring the Question Difficulty Effect | SIGIR | 2022 |
| [QIKT](https://arxiv.org/abs/2302.06885) | Improving Interpretability of Deep Sequential Knowledge Tracing Models with Question-centric Cognitive Representations | AAAI | 2023 |

- A summary of information for the selected datasets

| **Dataset** | **Questions** | **Knowledge Concepts** | **Students** | **Answers** | **Timestamp** | **Purpose** | **Questions Bundling** |
|-------------|---------------|------------------------|--------------|-------------|---------------|-------------|------------------------|
| [ASSIST12](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) | 45,716 | 265 | 27,066 | 2,541,201 | ✓ | Educational research | ✓ |
| [ASSIST17](https://sites.google.com/view/assistmentsdatamining/dataset?authuser=0) | 3,162 | 102 | 1,709 | 942,816 | ✓ | Educational longitudinal prediction | ✓ |
| [BA06](https://pslcdatashop.web.cmu.edu/KDDCup/) | 207,856 | 493 | 1,146 | 3,656,871 | ✓ | Mathematics-focused educational prediction |  |
| [NIPS34](https://eedi.com/projects/neurips-education-challenge) | 948 | 57 | 4,918 | 1,382,727 | ✓ | Educational research on diagnostic questions |  |
| [EN1](https://github.com/riiid/ednet) | 13,169 | 188 | 784,309 | 95,293,926 | ✓ | AI tutoring | ✓ |
