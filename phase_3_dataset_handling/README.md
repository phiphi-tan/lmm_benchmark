# Dataset Handling
> Last Updated: 03 June 2025
This phase involves automating annotation and filtering of datasets.

Additionally, there can also be further checks implemented to help increase the robustness of datasets used for testing.

## Post-processing

### Incomplete / Irrelevant Data
Many datasets provided will be either incomplete / contain unnecessary data, and thus some form of pre-filtering prior to testing is required to ensure that the final datasets obtained are suitable for our target problem.

#### Data Filtering Approaches
1. Manual Filtering


### Data contamination
Given that these papers are publicly available, there is a possibility that the models that are being benchmarked will have been introduced to some of the benchmark data at any point during their training, which may cause inaccuracy during benchmarks.

This brings us to the need to augment the input data to increase its robustness

####  Data augmentation approaches
