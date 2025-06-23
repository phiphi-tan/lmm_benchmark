from ..util.benchmark_tools import run_benchmark
from ..util.displays import show_individual, show_differences, show_results
from ..util.benchmark_models import get_models
from datasets import load_dataset

#----- hyperparameters -----

models = get_models()

dataset_path = "vikhyatk/tallyqa-test"
dataset_split = "test"
sample_size = 3
data_info = [dataset_path, dataset_split, sample_size]

system_prompt = "You are an object counting tool, and you can ONLY reply in numbers "\
"Analyze the image and respond with ONLY a single number. Do not provide any explanation or introductory text."
global_user_prompt = None
sys_user_prompt = [system_prompt, global_user_prompt]

metric_type = "exact_match"

#----- data preparation function -----

# must return input images (PIL), question list, reference list
def prep_data(ds_path, ds_split, split_size=None):
    print("Preparing data with size: {}".format(split_size))
    ds = load_dataset(ds_path, split=ds_split)

    print("Original Dataset: {}".format(ds))
    # filter for simple questions
    ds = ds.filter(lambda row: row["qa"][0]["is_simple"])
    print("Filtered Dataset: {}".format(ds))

    if split_size is not None:
        ds = ds.shuffle() # for random selection
        input_dataset = ds.select(range(split_size))
    else: 
        input_dataset = ds

    print("Sampled Dataset: {}".format(ds))

    image_list = input_dataset['image']

    qa_list = input_dataset['qa']
    question_list = [q[0]["question"] for q in qa_list]
    answer_list = [q[0]["answer"] for q in qa_list]

    return image_list, question_list, answer_list 

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

