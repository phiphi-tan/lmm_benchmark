from util.benchmark_tools import run_benchmark, show_individual, show_results
import util.benchmark_models as benchmark_models
from datasets import load_dataset
from PIL import Image
import io
import base64

#----- hyperparameters -----

models = benchmark_models.get_models()

dataset_path = "moondream/TallyQA-VLMEvalKit"
dataset_split = "train"
sample_size = 3

system_prompt = "You are an object counting tool. Your task is to estimate the number of objects in the provided image. "\
"Analyze the image and respond ONLY with a single number (no full stops). Do not provide any explanation or introductory text or punctuation."
global_user_prompt = None
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

    image_list = input_dataset['image']
    image_list = [base64.b64decode(img_str) for img_str in image_list]
    image_list = [Image.open(io.BytesIO(b)) for b in image_list]

    question_list = input_dataset['question']

    answer_list = input_dataset['answer']
    answer_list = [str(ans) for ans in answer_list]

    return image_list, question_list, answer_list 

#----- benchmarks -----
# DO NOT EDIT


inputs = prep_data(ds_path=dataset_path, ds_split=dataset_split, split_size=sample_size)
predictions, evaluations = run_benchmark(models=models, inputs=inputs, sys_user_prompts=[system_prompt, global_user_prompt], metric_type=metric_type)
# show_individual(inputs, predictions)
show_results(inputs, predictions, evaluations)

