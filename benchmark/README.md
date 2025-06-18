# Benchmark Scripts

This directory contains the scripts to run some basic benchmarks that relate to the 4 related subtasks (as stated in the [data requirements document](../data_requirements.md)).

## General Tools

### `benchmark_tools.py`

- Handles the internal benchmarking process (e.g. generating predictions, evaluation of the predictions)

### `benchmark_template.py`

- Basic template for running a benchmark on an "image-text-to-text" pipeline
- Easily duplicated for other tasks
- Requires the implementation of `prep_data(ds_path, ds_split, split_size=None)`
  - returns the list of images, question list (if any) and the reference list

## Benchmarking Process

### Models

Due to the huggingface pipeline architecture used, there are currently a limited amount of models that can be automatically tested. I am also currently using smaller sized models (0.5 - 3B instructions) due to hardware constraints. I am certain that performance will increase with larger models

> All models can be found in `util/benchmark_models.py`

```py
models = [
            "google/gemma-3-12b-it",
            "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "google/gemma-3-4b-it",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2-VL-2B-Instruct",
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            "HuggingFaceTB/SmolVLM-256M-Instruct",
          ]
```

### Dataset Preparation

Each file contains a dataset prepared for a certain task. Given that the structure of datasets are non-consistent, I found it necessary to include a `prep_data()` function in the template to allow users to structure their data in a standardised foramat (`image_list, question_data_list, answer_data_list`) that can be passed into the benchmark tool.

### Benchmarking Results

The detailed list of benchmarks obtained so far can be viewed [here](benchmark_results.md)

## Prompt Approaches

The model outputs are notoriously difficult to control, so part of ensuring an adequate benchmarking performance involves experimenting with different prompt approaches (i.e. 'prompt engineering').

Since models are trained on different bases, the more 'chatty' models are harder to prompt to provide just the answer. Similar to performance, I believe that utilising 'better' models trained on more instructions greatly increases the chances of generated output meeting required data.

At a basic level, a prompt is considered 'successful' if it results in the model providing output in the desired structure. Here are some approaches I've tried

- System-user prompts

  - Thanks to the HuggingFace chat template, I can structure prompts given to the model as either via 'system' or 'user', which allows me to pass overarching commands to a model while still providing a means for independent, data-related questions (e.g. Object Counting: System-level instructs the model on the task, user-level will ask "how many dogs are there")

- Completion-prompts
