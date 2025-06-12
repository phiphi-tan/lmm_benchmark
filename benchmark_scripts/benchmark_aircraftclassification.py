from benchmark_tools import run_benchmark, eval_results
from datasets import load_dataset
from PIL import Image
import io

#----- hyperparameters -----

model_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
dataset_path = "Mr-Fox-h/Civil_or_Military"
dataset_split = "train"
sample_size = 3

system_prompt = "You are a military aircraft classification tool. Your ONLY function is to classify aircraft images as either civilian (0) or military (1)."\
"Return ONLY the single digit classification without ANY additional text, explanation, or formatting.\n"
user_prompt = "Classify this aircraft image. Output ONLY a single digit: 0 for civilian aircraft or 1 for military aircraft. Do not include any other text or explanation."

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

image_list, question_list, reference_list = prep_data(ds_path=dataset_path,
                                                      ds_split=dataset_split,
                                                      split_size=sample_size)
print("image_list: {}".format(image_list))
print("question_list: {}".format(question_list))
print("reference_list: {}".format(reference_list))

prediction_list = run_benchmark(model=model_name,
                                img_list=image_list, qn_list=question_list,
                                data_size=sample_size, sys_prompt=system_prompt, global_user_prompt=user_prompt)

print("prediction_list: {}".format(prediction_list))

eval_results(img_list=image_list, qn_list=question_list, pred_list=prediction_list, ref_list=reference_list,
            data_size=sample_size, metric_type=metric_type)
