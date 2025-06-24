from transformers import pipeline
import evaluate
from .judge import judge_captions
from .bounding_boxes import eval_bbox
import torch

# split inputs according to prep_data output
def split_inputs(inputs):
    print("Image list: {}".format(inputs[0]))
    print("Question list: {}".format(inputs[1]))
    print("Reference list: {}".format(inputs[2]))

    return inputs[0], inputs[1], inputs[2]

# split prompts into system and user prompts
def split_prompts(sys_user_prompts):
    return sys_user_prompts[0], sys_user_prompts[1]

# runs get_predictions and eval_results; returns predictions and evaluations
def run_benchmark(prep_data, data_info, models, sys_user_prompts, metric_type, edit_predictions=None):

    dataset_path, dataset_split, sample_size = data_info
    inputs = prep_data(ds_path=dataset_path, ds_split=dataset_split, split_size=sample_size)

    num_of_models = len(models)
    predictions = {}
    model_results = {}
    
    sys_prompt, user_prompt = split_prompts(sys_user_prompts)
    img_list, qn_list, ref_list = split_inputs(inputs)

    for i in range(num_of_models):
        model = models[i]
        prediction_list = get_predictions(model=model, img_list=img_list, qn_list=qn_list,
                                          sys_prompt=sys_prompt, global_user_prompt=user_prompt)
        predictions[model] = prediction_list
        print("prediction_list ({}): {}".format(model, prediction_list))

    if edit_predictions is not None:
        predictions = edit_predictions(predictions)

    for model in predictions:
        prediction_list = predictions[model]
        if metric_type == "llm_aaj":
            evaluation = judge_captions(model, img_list, ref_list, prediction_list)
        if metric_type == "bbox":
            evaluation = eval_bbox(ref_list, img_list, prediction_list)
        else:
            evaluation = eval_results(ref_list=ref_list,pred_list=prediction_list, metric_type=metric_type)
            
        model_results[model] = evaluation
        print("evaluation ({}): {}".format(model, evaluation))

    return inputs, predictions, model_results

# returns output_list
def get_predictions(model, img_list, qn_list, sys_prompt, global_user_prompt=None):
    print("Obtaining predictions from {}".format(model))
    pipe = pipeline("image-text-to-text", model=model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",)
    pipe.model.config.pad_token_id = pipe.tokenizer.eos_token_id

    data_size = len(img_list)
    prediction_list = [None for _ in range(data_size)]

    for i in range(data_size):
        print("Predicting sample {} of {}".format(i + 1, data_size))
        img = img_list[i]
        # take prompt from qn_list is there is no global user_prompt
        if global_user_prompt is None: 
            user_prompt = qn_list[i]
        else:
            user_prompt = global_user_prompt

        messages = [
            {"role": "system",
            "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
        ]

        output = pipe(images=img,
                    text=messages,
                    generate_kwargs={"max_new_tokens": 50},
                    return_full_text=False)

        prediction = output[0]['generated_text']
        prediction_list[i] = prediction

    return prediction_list

# returns evaluation result
def eval_results(ref_list, pred_list, metric_type):
    print("Evaluating predictions")
    eval_metric = evaluate.load(metric_type)
    results = eval_metric.compute(references=ref_list, predictions=pred_list)
    return_result = round(results[metric_type], 2)
    return return_result

