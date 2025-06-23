from ..util.benchmark_tools import run_benchmark
from ..util.displays import show_individual, show_results
from ..util.benchmark_models import get_models
from datasets import load_dataset

#----- hyperparameters -----
models = get_models()

dataset_path = "Mr-Fox-h/Civil_or_Military"
dataset_split = "train"
sample_size = 3
data_info = [dataset_path, dataset_split, sample_size]

system_prompt = "You are a military aircraft classification tool. Your ONLY function is to classify aircraft images as either civilian (0) or military (1)."\
"Return ONLY the single digit classification without ANY additional text, explanation, or formatting.\n"
global_user_prompt = "Classify this aircraft image. Output ONLY a single digit: 0 for civilian aircraft or 1 for military aircraft.  Give the shortest answer possible. Do not include any other text or explanation."
sys_user_prompt = [system_prompt, global_user_prompt]

metric_type = "exact_match"

#----- data preparation function -----

# must return input images (PIL), question list, reference list
def prep_data(ds_path, ds_split, split_size=None):
    ds = load_dataset(ds_path, split=ds_split)

    if split_size is not None:
        shuffled_ds = ds.shuffle() # for random selection
        input_dataset = shuffled_ds.select(range(split_size))
    else: 
        input_dataset = ds

    image_list = input_dataset['image'] # get list of images
    question_data_list = [] # get list of questions (if necessary)
    answer_data_list = input_dataset['label'] # get list of answers
    answer_data_list = [str(i) for i in answer_data_list] # change to list of strings for comparison

    return image_list, question_data_list, answer_data_list 

def edit_predictions(predictions):
    new_predictions = predictions.copy()

    for model in predictions:
        if model != 'HuggingFaceTB/SmolVLM-256M-Instruct': continue
        
        pred = predictions[model]
        new_predictions[model+' (edited)'] = [p.replace(".", "").strip() for p in pred]
    return new_predictions

#----- benchmarks -----
# DO NOT EDIT

inputs, predictions, evaluations = run_benchmark(prep_data=prep_data, data_info=data_info,
                                                models=models, sys_user_prompts=sys_user_prompt,
                                                edit_predictions=edit_predictions, metric_type=metric_type)
# show_individual(inputs, predictions)
show_results(inputs, predictions, evaluations)




    