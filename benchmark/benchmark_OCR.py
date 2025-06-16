from util.benchmark_tools import run_benchmark, show_individual, show_results
import util.benchmark_models as benchmark_models
from datasets import load_dataset

#----- hyperparameters -----
models = benchmark_models.get_models()

dataset_path = "MiXaiLL76/TextOCR_OCR"
dataset_split = "train"
sample_size = 10

system_prompt = "You are an optical character recognition (OCR) tool. Your ONLY function is to process an input image and output the text shown."\
"Do not provide any explanation or introductory text."
global_user_prompt = "Analyze the image and respond with ONLY the text shown." # set to None if passing individual prompts
metric_type = "exact_match"
#----- data preparation function -----

# must return input images (PIL), question list, reference list
def prep_data(ds_path, ds_split, split_size=None):
    ds = load_dataset(ds_path, split=ds_split, streaming=True)

    if split_size is not None:
        shuffled_ds = ds.shuffle() # for random selection
        input_dataset = shuffled_ds.select(range(split_size))
    else: 
        input_dataset = ds

    image_list = input_dataset['image'] # get list of images
    question_data_list = [] # get list of questions (if necessary)
    answer_data_list = input_dataset['text'] # get list of answers

    return image_list, question_data_list, answer_data_list 

#----- benchmarks -----
# DO NOT EDIT

inputs = prep_data(ds_path=dataset_path, ds_split=dataset_split, split_size=sample_size)
predictions, evaluations = run_benchmark(models=models, inputs=inputs, sys_user_prompts=[system_prompt, global_user_prompt], metric_type=metric_type)

# show_individual(inputs, predictions)
show_results(inputs, predictions, evaluations)

