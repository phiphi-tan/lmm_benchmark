from util.benchmark_tools import run_benchmark
from util.displays import show_individual, show_results
import util.benchmark_models as benchmark_models
from datasets import load_dataset

#----- hyperparameters -----
models = benchmark_models.get_models()

dataset_path = "howard-hou/OCR-VQA"
dataset_split = "train"
sample_size = 5
data_info = [dataset_path, dataset_split, sample_size]

system_prompt = "You are an optical character recognition (OCR) tool. Output ONLY the exact answer to the question about the input image." \
"Answer briefly -- give the shortest answer possible (a word or phrase) with no explanatory text."
global_user_prompt = None # set to None if passing individual prompts
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

    image_list = input_dataset['image'] # get list of images

    question_data_list = input_dataset['questions'] # get list of questions (if necessary)
    question_data_list = [q[0] for q in question_data_list]

    answer_data_list = input_dataset['answers'] # get list of answers
    answer_data_list = [a[0] for a in answer_data_list]

    return image_list, question_data_list, answer_data_list 

def edit_predictions(predictions):
    new_predictions = predictions.copy()

    for model in predictions:
        if model != 'HuggingFaceTB/SmolVLM-256M-Instruct': continue

        pred = predictions[model]
        # remove trails and last character if it is a full stop
        edited_list = [p.strip() for p in pred]
        edited_list = [p[:-1] if p.endswith('.') else p for p in edited_list]
        new_predictions[model+' (edited)'] = edited_list
        
    return new_predictions

#----- benchmarks -----
# DO NOT EDIT

inputs, predictions, evaluations = run_benchmark(prep_data=prep_data, data_info=data_info,
                                                models=models, sys_user_prompts=sys_user_prompt,
                                                edit_predictions=edit_predictions, metric_type=metric_type)
# show_individual(inputs, predictions)
show_results(inputs, predictions, evaluations)

