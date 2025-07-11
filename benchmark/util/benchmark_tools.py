from transformers import pipeline
import evaluate
from .judge import judge_captions
from .bounding_boxes import eval_bbox
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

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
        if model == "Qwen/Qwen2.5-Omni-7B" or model == "Qwen/Qwen2.5-Omni-3B":
            prediction_list = get_omni_predictions(model, img_list, qn_list, sys_prompt, user_prompt)
        else:
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
        if metric_type == "bbox_iou":
            evaluation = eval_bbox(ref_list, prediction_list, img_list=img_list, normalise=True)
        else:
            evaluation = eval_results(ref_list=ref_list,pred_list=prediction_list, metric_type=metric_type)
            
        model_results[model] = evaluation
        print("evaluation ({}): {}".format(model, evaluation))

    return inputs, predictions, model_results

def get_omni_predictions(model_name, img_list, qn_list, sys_prompt, global_user_prompt=None):
    print("Obtaining predictions from {}".format(model_name))
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    model.disable_talker()

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
            "content": [{"type": "image", "image": img},
                        {"type": "text", "text": user_prompt}]},
        ]

        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        _, images, _ = process_mm_info(messages, use_audio_in_video=False)
        inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        input_length = inputs.input_ids.shape[1]

        # Inference: Generation of the output text and audio
        text_ids = model.generate(**inputs, return_audio=False, max_new_tokens=15)
        prediction = processor.batch_decode(text_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(prediction)
        prediction_list[i] = prediction

    return prediction_list


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
                    generate_kwargs={"max_new_tokens": 30},
                    return_full_text=False)

        prediction = output[0]['generated_text']
        prediction_list[i] = prediction

    return prediction_list

# returns evaluation result
def eval_results(ref_list, pred_list, metric_type):
    print("Evaluating predictions")
    eval_metric = evaluate.load(metric_type)
    results = eval_metric.compute(references=ref_list, predictions=pred_list)
    return_result = round(results[metric_type], 3)
    return return_result

