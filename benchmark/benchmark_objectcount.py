from util.benchmark_tools import run_benchmark
import util.benchmark_models as benchmark_models
from datasets import load_dataset
from PIL import Image
import io

#----- hyperparameters -----

models = benchmark_models.get_models()

dataset_path = "nimapourjafar/mm_tallyqa"
dataset_split = "train"
sample_size = 3

system_prompt = "You are an object counting tool. Your task is to estimate the number of objects in the provided image. "\
"Analyze the image and respond ONLY with a single number (no full stops). Do not provide any explanation or introductory text or punctuation."
global_user_prompt = None
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

    image_list = input_dataset['images']
    image_list = [img[0]['bytes'] for img in image_list]
    image_list = [Image.open(io.BytesIO(b)) for b in image_list]

    data_list = input_dataset['data']
    question_data_list = [x[1] for x in data_list] # get data of first question
    question_data_list = [q['data'] for q in question_data_list] # get first question as string
    answer_data_list = [x[2] for x in data_list] # get data of answer
    answer_data_list = [q['data'] for q in answer_data_list] # get answer as string
    answer_data_list = [s[:-1] for s in answer_data_list] # remove the '.' at the end of the answers

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

