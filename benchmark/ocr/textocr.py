from ..util.benchmark_tools import run_benchmark
from ..util.displays import show_individual, show_results
from ..util.benchmark_models import get_models
from datasets import load_dataset

#----- hyperparameters -----
models = get_models()

dataset_path = "MiXaiLL76/TextOCR_OCR"
dataset_split = "train"
sample_size = 64
data_info = [dataset_path, dataset_split, sample_size]

system_prompt = "You are an optical character recognition (OCR) tool. Your ONLY function is to process an input image and output the text shown."\
"Do not provide any explanation or introductory text."
global_user_prompt = "Analyze the image and respond with ONLY the text shown. Give the shortest answer possible." # set to None if passing individual prompts
sys_user_prompt = [system_prompt, global_user_prompt]

metric_type = "exact_match"
#----- data preparation function -----

# must return input images (PIL), question list, reference list
def prep_data(ds_path, ds_split, split_size=None):
    print("Preparing data with size: {}".format(split_size))
    ds = load_dataset(ds_path, split=ds_split)

    if split_size is not None:
        # ds = ds.shuffle(seed=split_size) # for random selection
        input_dataset = ds.select(range(split_size))
    else: 
        input_dataset = ds

    image_list = input_dataset['image'] # get list of images
    question_data_list = [] # get list of questions (if necessary)
    answer_data_list = input_dataset['text'] # get list of answers

    return image_list, question_data_list, answer_data_list 

def edit_predictions(predictions):
    new_predictions = predictions.copy()

    for model in predictions:
        if model != 'HuggingFaceTB/SmolVLM-256M-Instruct': continue

        pred = predictions[model]
        new_predictions[model+' (edited)'] = [p.strip() for p in pred]
    return new_predictions

#----- benchmarks -----
# DO NOT EDIT

inputs, predictions, evaluations = run_benchmark(prep_data=prep_data, data_info=data_info,
                                                models=models, sys_user_prompts=sys_user_prompt,
                                                edit_predictions=edit_predictions, metric_type=metric_type)
# show_individual(inputs, predictions)
show_results(inputs, predictions, evaluations)

