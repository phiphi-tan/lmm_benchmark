from ..util.benchmark_tools import run_benchmark
from ..util.displays import show_individual, show_differences, show_results
from ..util.benchmark_models import get_models
from datasets import load_dataset

#----- hyperparameters -----

models = get_models()

dataset_path = "Naveengo/flickr8k"
dataset_split = "train"
sample_size = 64
data_info = [dataset_path, dataset_split, sample_size]

system_prompt = "You are an image caption tool. Your task is generate captions for a given input image."\
"Do not provide any explanation or introductory text, or anything other than what is shown."
global_user_prompt = "Analyze the image and respond ONLY with a caption."
sys_user_prompt = [system_prompt, global_user_prompt]

metric_type = "llm_aaj"

#----- data preparation function -----

# must return input images (PIL), question list, reference list
def prep_data(ds_path, ds_split, split_size=None):
    ds = load_dataset(ds_path, split=ds_split)

    if split_size is not None:
        shuffled_ds = ds.shuffle(seed=split_size) # for random selection
        input_dataset = shuffled_ds.select(range(split_size))
    else: 
        input_dataset = ds

    image_list = input_dataset['image'] # get list of images
    question_data_list = [] # get list of questions (if necessary)
    answer_data_list = input_dataset['text'] # get list of answers

    return image_list, question_data_list, answer_data_list 

def edit_predictions(predictions):
    new_predictions = predictions.copy()

    raise NotImplementedError
    for model in predictions:
        pred = predictions[model]
    return new_predictions


#----- benchmarks -----
# DO NOT EDIT

inputs, predictions, evaluations = run_benchmark(prep_data=prep_data, data_info=data_info,
                                                models=models, sys_user_prompts=sys_user_prompt,
                                                metric_type=metric_type)
show_results(inputs, predictions, evaluations)
# show_individual(inputs, predictions, judge_evaluations=evaluations)

    