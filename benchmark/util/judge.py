from transformers import pipeline

# two-pass, returns both the scores as well as reasoning
def judge_captions(model, img_list, ref_list, cand_list):
    return 0
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