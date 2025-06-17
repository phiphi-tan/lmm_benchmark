from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import evaluate

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
    pipe = pipeline("image-text-to-text", model=model, trust_remote_code=True)
    pipe.model.config.pad_token_id = pipe.tokenizer.eos_token_id

    data_size = len(img_list)
    prediction_list = [None for _ in range(data_size)]

    for i in range(data_size):
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

# two-pass, returns both the scores as well as reasoning
def judge_captions(model, img_list, ref_list, cand_list):

    data_size = len(img_list)

    judge_prompt = "Your task is to evaluate and rate the candidate caption on a scale of 0.0 to 1.0 based on the given Grading Criteria. " \
        "(Print ONLY a decimal number between 0.0 and 1.0) " \
        "Grading Scale:" \
        "0.0: The caption does not describe the image at all." \
        "0.5: The caption is partially correct but has notable inaccuracies." \
        "1.0: The caption accurately and clearly describes the image." \
        "Examples: - Perfect match: 1.0 - One minor detail wrong: 0.8 - Missing half the content: 0.5 - Completely wrong: 0.0\n" \
        "Reference Captions: {ref}" \
        "Candidate Caption: {cand}" \
        "Score (Print ONLY the decimal number): "

    judge_followup_prompt = "Why? Tell me the reason."

    pipe = pipeline("image-text-to-text", model=model)
    pipe.model.config.pad_token_id = pipe.tokenizer.eos_token_id

    score_list = [None for _ in range(data_size)]
    reason_list = [None for _ in range(data_size)]

    for i in range(data_size):
        input_prompt = judge_prompt.format(ref=ref_list[i], cand=cand_list[i])

        msg = [
            {"role": "user",
            "content": [{"type": "image"},
                        {"type": "text", "text": input_prompt}]
            },
        ]

        output = pipe(images=img_list[i],
                text=msg,
                generate_kwargs={"max_new_tokens": 10},
                return_full_text=False)
        
        score = output[0]['generated_text']
        score_list[i] = score
        
        followup_msg = [
                {"role": "user",
                "content": [{"type": "image"},
                            {"type": "text", "text": input_prompt}]
                },
                {"role": "assistant",
                "content": [{"type": "text", "text": score}]},
                {"role": "user",
                "content": [{"type": "text", "text": judge_followup_prompt}]},
            ]

        reason_output = pipe(images=img_list[i],
                            text=followup_msg,
                            generate_kwargs={"max_new_tokens": 50},
                            return_full_text=False)
    
        judge_reason = reason_output[0]['generated_text']
        reason_list[i] = judge_reason
                       
    return score_list, reason_list

# for Colab
def show_individual(inputs, predictions, judge_evaluations=None):
    img_list, qn_list, ref_list = split_inputs(inputs)
    if len(qn_list) == 0: # only global questions, no individual
        qn_list = ["" for _ in img_list]

    for i in range(len(img_list)):
        display(img_list[i])
        print("Question: {}".format(qn_list[i]))
        print("Reference: {}".format(ref_list[i]))
        for key, val in predictions.items():
            print("Predicted ({}): {}".format(key, val[i]))
            if judge_evaluations is not None:
                judge_scores = judge_evaluations[key][0]
                judge_reasons = judge_evaluations[key][1]
                print("Rating ({}): {} ({})".format(key, judge_scores[i], judge_reasons[i]))

def show_results(inputs, predictions, evaluations):
    _, _, ref_list = split_inputs(inputs)
    print("Benchmark Results:")
    print("Truth: {}".format(ref_list))
    for key, val in evaluations.items():
        print("{}: {} ({})".format(key, val, predictions[key]))

def draw_bboxes(image, bbox_list, colour='red'):
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    for bbox in bbox_list:
        draw.rectangle(bbox, outline=colour, width=3)
    return new_image

# when coordinates must be ordered (smaller first)
def fix_bbox(bbox):
    x1, y1, x2, y2 = bbox

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y2
    
    return [x1, y1, x2, y2]

def draw_bboxes(image, bbox_list):
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    for bbox in bbox_list:
        bbox = fix_bbox(bbox)
        draw.rectangle(bbox, outline='red', width=3)
    return new_image

def draw_bbox_normalised(image, bbox):
    new_image = image.copy()
    img_width, img_height = new_image.size
    draw = ImageDraw.Draw(new_image)
    bbox = fix_bbox(bbox)
    new_bbox = [bbox[0]*img_width, bbox[1]*img_height, bbox[2]*img_width, bbox[3]*img_height]
    draw.rectangle(new_bbox, outline='blue', width=3)

    return new_image

def show_differences(inputs, predictions):
    img_list, _, ref_list = split_inputs(inputs)
    for i in range(len(img_list)):
        new_img = draw_bboxes(img_list[i], ref_list[i])

        for key, val in predictions.items():
            # print("pred ({}) is {}".format(key, val))
            new_img = draw_bbox_normalised(new_img, val[i])
            display(new_img)

def eval_bbox(ref_list, img_list, pred_list):
    eval = []
    for i in range(len(ref_list)):
        ref_bbox = ref_list[i][0]
        ref_bbox = fix_bbox(ref_bbox)

        img = img_list[i]
        img_width, img_height = img.size

        pred_bbox = pred_list[i] # standardised values
        pred_bbox = [pred_bbox[0]*img_width, pred_bbox[1]*img_height, pred_bbox[2]*img_width, pred_bbox[3]*img_height]
        pred_bbox = fix_bbox(pred_bbox)

        iou = intersection_over_union(ref_bbox, pred_bbox)
        iou = round(iou, 2)
        eval.append(iou)
    
    return eval

def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    # The width and height must be positive, so we use max(0, ...)
    # to handle cases where there is no overlap.
    intersection_width = max(0, xB - xA)
    intersection_height = max(0, yB - yA)
    intersection_area = intersection_width * intersection_height

    # Compute the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the area of the union: area(A) + area(B) - area(intersection)
    union_area = float(boxA_area + boxB_area - intersection_area)

    # Compute the intersection over union
    # Handle the case of division by zero if union_area is 0
    iou = intersection_area / union_area if union_area != 0 else 0

    return iou