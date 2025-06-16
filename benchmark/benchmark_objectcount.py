from util.benchmark_tools import run_benchmark, show_individual, show_results
import util.benchmark_models as benchmark_models
from datasets import load_dataset
from PIL import Image
import io

#----- hyperparameters -----

models = benchmark_models.get_models() # remove for Colab

dataset_path = "nimapourjafar/mm_tallyqa"
dataset_split = "train"
sample_size = 3
data_info = [dataset_path, dataset_split, sample_size]

system_prompt = "You are an object counting tool. Your task is to estimate the number of objects in the provided image. "\
"Analyze the image and respond ONLY with a single number. Do not provide any explanation or introductory text or punctuation."
global_user_prompt = None
sys_user_prompt = [system_prompt, global_user_prompt]

metric_type = "exact_match"

#----- data preparation function -----

# must return input images (PIL), question list, reference list
def prep_data(ds_path, ds_split, split_size=None):
    print("Preparing data with size: {}".format(split_size))
    ds = load_dataset(ds_path, split=ds_split)

    if split_size is not None:
        shuffled_ds = ds.shuffle() # for random selection
        input_dataset = shuffled_ds.select(range(split_size))
    else: 
        input_dataset = ds

    image_list = input_dataset['images']
    image_list = [img[0]['bytes'] for img in image_list]
    image_list = [Image.open(io.BytesIO(b)) for b in image_list]

    data_list = input_dataset['data']
    question_data_list = [x[1] for x in data_list]
    question_data_list = [q['data'] for q in question_data_list]
    answer_data_list = [x[2] for x in data_list]
    answer_data_list = [q['data'] for q in answer_data_list]
    answer_data_list = [s[:-1] for s in answer_data_list]

    return image_list, question_data_list, answer_data_list 

# change if the output from the model needs to be edited
def edit_predictions(predictions):
    new_predictions = predictions.copy()

    for model in predictions:
        pred = predictions[model]
        # REMOVE punctuation coz models dont follow instructions
        # Remove trailing / leading spaces
        if model != 'HuggingFaceTB/SmolVLM-256M-Instruct': continue

        new_predictions[model+' (edited)'] = [p.replace(".", "").strip() for p in pred]
    return new_predictions

#----- benchmarks call -----
# DO NOT EDIT

inputs, predictions, evaluations = run_benchmark(prep_data=prep_data, data_info=data_info,
                                                models=models, sys_user_prompts=sys_user_prompt,
                                                edit_predictions=edit_predictions, metric_type=metric_type)
# show_individual(inputs, predictions)
show_results(inputs, predictions, evaluations)

