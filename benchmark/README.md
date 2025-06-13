# Benchmark Scripts

This directory contains the scripts to run some basic scripts that relate to the 4 related subtasks (as stated in the [data requirements document](../data_requirements.md)).

## General Tools

### `benchmark_tools.py`

- Handles the internal benchmarking process (e.g. generating predictions, evaluation of the predictions)

### `benchmark_template.py`

- Basic template for running a benchmark on an "image-text-to-text" pipeline
- Easily duplicated for other tasks
- Requires the implementation of `prep_data(ds_path, ds_split, split_size=None)`
  - returns the list of images, question list (if any) and the reference list

## Benchmarking Process

### Hyperparameters

### Data Preparation

### Benchmarking
