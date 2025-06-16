from transformers import pipeline
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
def run_benchmark(models, inputs, sys_user_prompts, metric_type):
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

        if metric_type == "llm_aaj":
            evaluation = judge_captions(model, img_list, ref_list, prediction_list)
        else:
            evaluation = eval_results(img_list=img_list, qn_list=qn_list, ref_list=ref_list,
                                  pred_list=prediction_list, metric_type=metric_type)
            
        model_results[model] = evaluation
        print("evaluation ({}): {}".format(model, evaluation))

    return predictions, model_results



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
                    generate_kwargs={"max_new_tokens": 10},
                    return_full_text=False)

        prediction = output[0]['generated_text']
        
        # REMOVE punctuation coz models dont follow instructions
        # Remove trailing / leading spaces
        prediction = prediction.replace(".","").strip()

        prediction_list[i] = prediction

    return prediction_list

# returns evaluation result
def eval_results(img_list, qn_list, ref_list, pred_list, metric_type, breakdown=False):
    print("Evaluating predictions")
    eval_metric = evaluate.load(metric_type)
    data_size = len(img_list)
    if breakdown is True:
        print("Results of the benchmark:")

        for i in range(data_size):
            img_list[i].show() # for local machine
            # display(img_list[i]) # for Google Colab
            print("Question: {}".format(qn_list[i]))
            print("Predicted: {}".format(pred_list[i]))
            print("Actual: {}".format(ref_list[i]))

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
                "content": [{"type": "text", "text": score + "/ 1.0"}]},
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
def show_individual(inputs, predictions):
    img_list, qn_list, ref_list = split_inputs(inputs)
    if len(qn_list) == 0: # only global questions, no individual
        qn_list = ["" for _ in img_list]

    for i in range(len(img_list)):
        display(img_list[i])
        print("Question: {}".format(qn_list[i]))
        print("Truth: {}".format(ref_list[i]))
        for key, val in predictions.items():
            print("Predicted ({}): {}".format(key, val[i]))

def show_results(inputs, predictions, evaluations):
    _, _, ref_list = split_inputs(inputs)
    print("Benchmark Results:")
    print("Truth: {}".format(ref_list))
    for key, val in evaluations.items():
        print("{}: {} ({})".format(key, val, predictions[key]))