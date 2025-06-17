from util.benchmark_tools import run_benchmark, show_individual, show_results, draw_bboxes
import util.benchmark_models as benchmark_models
import ast
from datasets import load_dataset

#----- hyperparameters -----
models = benchmark_models.get_models()

dataset_path = "Illia56/Military-Aircraft-Detection"
dataset_split = "train"
sample_size = 5
data_info = [dataset_path, dataset_split, sample_size]

system_prompt = "You are a weapon detection tool tool. Your ONLY function is to provide the coordinates of detected weapons in a corner-coordinates bounding box format."\
"Do not provide any explanation or introductory text."
global_user_prompt = "Analyze the image and respond with ONLY the corner-coordinates of the weapon in square brackets ONLY. Give the shortest answer possible." # set to None if passing individual prompts
sys_user_prompt = [system_prompt, global_user_prompt]

metric_type = "bbox"

#----- data preparation function -----

# must return input images (PIL), question list, reference list
def prep_data(ds_path, ds_split, split_size=None):
    print("Preparing data with size: {}".format(split_size))
    ds = load_dataset(ds_path, split=ds_split)
    input_dataset = None

    if split_size is not None:
        shuffled_ds = ds.shuffle() # for random selection
        input_dataset = shuffled_ds.select(range(split_size))
    else: 
        input_dataset = ds
    print(input_dataset)
    image_list = input_dataset['image'] # get list of images
    question_data_list = [] # get list of questions (if necessary)
    ref_data_list = input_dataset['objects'] # get list of answers
    ref_data_list = [r['bbox'] for r in ref_data_list]

    print(input_dataset)
    print(image_list)
    print(ref_data_list)

    return image_list, question_data_list, ref_data_list 

def edit_predictions(predictions):
    new_predictions = predictions.copy()

    for model in predictions:
        pred = predictions[model]
        print(pred)

        pred = [list(ast.literal_eval(p)) for p in pred]
        print(pred)
        new_predictions[model] = pred

    return new_predictions

#----- benchmarks -----
# DO NOT EDIT

inputs, predictions, evaluations = run_benchmark(prep_data=prep_data, data_info=data_info,
                                                models=models, sys_user_prompts=sys_user_prompt,
                                                edit_predictions=edit_predictions, metric_type=metric_type)
# show_differences(inputs, predictions)
show_results(inputs, predictions, evaluations)

    
