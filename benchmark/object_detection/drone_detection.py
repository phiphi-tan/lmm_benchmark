from ..util.benchmark_tools import run_benchmark
from ..util.displays import show_individual, show_differences, show_results
from ..util.benchmark_models import get_models
from ..util.bounding_boxes import fix_bbox, coco_bbox

import ast
from datasets import load_dataset, Dataset

#----- hyperparameters -----
models = get_models()

dataset_path = "pathikg/drone-detection-dataset"
dataset_split = "test"
sample_size = 64
data_info = [dataset_path, dataset_split, sample_size]

system_prompt = "You are a drone detection tool tool. Your ONLY function is to provide the coordinates of detected drones in a corner-coordinates bounding box format."\
"Do not provide any explanation or introductory text."
global_user_prompt = "Analyze the image and respond with ONLY the corner-coordinates of the drone in square brackets ONLY. Give the shortest answer possible." # set to None if passing individual prompts
sys_user_prompt = [system_prompt, global_user_prompt]

metric_type = "bbox_iou"

#----- data preparation function -----

# must return input images (PIL), question list, reference list
def prep_data(ds_path, ds_split, split_size):
    print("Preparing data with size: {}".format(split_size))
    ds = load_dataset(ds_path, split=ds_split)

    print("Original Dataset: {}".format(ds))
    # filter for only single-drone detection
    ds = ds.filter(lambda row: len(row['objects']['category']) == 1)
    print("Filtered Dataset: {}".format(ds))

    if split_size is not None:
        shuffled_ds = ds.shuffle(seed=split_size) # for random selection
        input_dataset = shuffled_ds.select(range(split_size))
    else: 
        input_dataset = ds


    image_list = input_dataset['image'] # get list of images
    question_data_list = [] # get list of questions (if necessary)
    ref_data_list = input_dataset['objects'] # get list of answers
    ref_data_list = [r['bbox'] for r in ref_data_list]

    print(input_dataset)
    print(image_list)
    print(ref_data_list)

    ref_data_list = [coco_bbox(bbox[0]) for bbox in ref_data_list]
    print(ref_data_list)

    return image_list, question_data_list, ref_data_list 

def edit_predictions(predictions):
    new_predictions = predictions.copy()

    for model in predictions:
        pred = predictions[model]
        print(pred)
        
        new_pred = []

        for p in pred:
            try:
                new_pred.append(ast.literal_eval(p))
            except:
                new_pred.append(p)
        
        print(new_pred)
        new_predictions[model] = new_pred

    return new_predictions

#----- benchmarks -----
# DO NOT EDIT

inputs, predictions, evaluations = run_benchmark(prep_data=prep_data, data_info=data_info,
                                                models=models, sys_user_prompts=sys_user_prompt,
                                                edit_predictions=edit_predictions, metric_type=metric_type)
show_results(inputs, predictions, evaluations)
# show_differences(inputs, predictions, input_normal=False)


    
