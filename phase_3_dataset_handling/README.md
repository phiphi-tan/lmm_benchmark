# Dataset Handling
This phase involves automating annotation and filtering of datasets.

Additionally, there can also be further checks implemented to help increase the robustness of datasets used for testing.

## Categorisation of Data
While the entire dataset is provided as a means to benchmark LMMs with regards to the provided tasks, the data has been split into 4 (non-exclusive) categories for each of the 4 tasks respectively.

Within each subtask, there may be additional nested sub-tasks where necessary to further categories the dataset and ensure that its scope covers the given requirements.

An overview of the subtask categories will be provided in the [data requirements document](../data_requirements.md)

## Pre-processing

### Incomplete / Irrelevant Data
Many datasets provided will be either incomplete / contain unnecessary data, and thus some form of pre-filtering prior to testing is required to ensure that the final datasets obtained are suitable for our target problem.

#### Data Filtering Approaches
1. Manual Filtering


### Data contamination
Given that these papers are publicly available, there is a possibility that the models that are being benchmarked will have been introduced to some of the benchmark data at any point during their training, which may cause inaccuracy during benchmarks.

This brings us to the need to augment the input data to increase its robustness

####  Data augmentation approaches



