from util.benchmark_tools import run_benchmark
import util.benchmark_models as benchmark_models
from datasets import load_dataset

#----- hyperparameters -----

models = benchmark_models.get_models()

dataset_path = "Mr-Fox-h/Civil_or_Military"
dataset_split = "train"
sample_size = 3

system_prompt = "You are a military aircraft classification tool. Your ONLY function is to classify aircraft images as either civilian (0) or military (1)."\
"Return ONLY the single digit classification without ANY additional text, explanation, or formatting.\n"
global_user_prompt = "Classify this aircraft image. Output ONLY a single digit: 0 for civilian aircraft or 1 for military aircraft. Do not include any other text or explanation."

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

#----- benchmarks -----
# DO NOT EDIT

inputs = prep_data(ds_path=dataset_path, ds_split=dataset_split, split_size=sample_size)
predictions, evaluations = run_benchmark(models=models, inputs=inputs, sys_user_prompts=[system_prompt, global_user_prompt], metric_type=metric_type)

print("question_list: {}".format(inputs[1]))
print("reference_list: {}".format(inputs[2]))

print("Benchmark Results:")
for key, val in evaluations.items():
    print("{}: {} ({})".format(key, predictions[key], val))
    



    