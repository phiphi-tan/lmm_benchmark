from util.benchmark_tools import run_benchmark, eval_results, judge_captions
from datasets import load_dataset
from PIL import Image
import io

#----- hyperparameters -----

model_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
dataset_path = "Naveengo/flickr8k"
dataset_split = "train"
sample_size = 1

system_prompt = "You are an image caption tool. Your task is generate captions for a given input image."\
"Do not provide any explanation or introductory text, or anything other than what is shown."
user_prompt = "Analyze the image and respond ONLY with a caption."



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
    answer_data_list = input_dataset['text'] # get list of answers

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

score_list, reason_list = judge_captions(model=model_name, img_list=image_list, cand_list=prediction_list, ref_list=reference_list,
                      data_size=sample_size)
print("score_list: {}".format(score_list))
print("reason_list: {}".format(reason_list))

eval_results(img_list=image_list, qn_list=question_list, pred_list=prediction_list, ref_list=reference_list,
            data_size=sample_size, metric_type=metric_type)
