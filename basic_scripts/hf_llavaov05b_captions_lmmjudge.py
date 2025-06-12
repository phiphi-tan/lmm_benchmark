# https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf
# https://huggingface.co/datasets/Naveengo/flickr8k

from transformers import pipeline
from datasets import load_dataset
from PIL import Image 


ds = load_dataset("Naveengo/flickr8k", split='train')
shuffled_ds = ds.shuffle()

num_samples_to_process = 3
ds_subset = shuffled_ds.select(range(num_samples_to_process))
print(ds_subset)

prompt = "You are an image caption tool. Your task is generate captions for a given input image."\
"Analyze the image and respond with an appropriate caption. Do not provide any explanation or introductory text, or anything other than what is shown."

judge_score_template = "Your task is to evaluate and rate the candidate caption on a scale of 0.0 to 1.0 based on the given Grading Criteria. " \
"(Print Real Number Score ONLY) " \
"Grading Criteria:" \
"0.0: The caption does not describe the image at all." \
"1.0: The caption accurately and clearly describes the image." \
"Reference Captions: {ref}" \
"Candidate Caption: {cand}" \
"Score (Choose a rating from 0.0 to 1.0): "

judge_followup_prompt = "Why? Tell me the reason."

pipe = pipeline("image-text-to-text", model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
pipe2 = pipeline("image-text-to-text", model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
pipe3 = pipeline("image-text-to-text", model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")

reference_list = ds_subset['text']

for i, entry in enumerate(ds_subset):
    im = entry['image']
    im.show()
    # display(im) # for Google Colab

    messages = [
        {"role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    output = pipe(images=entry['image'],
                  text=messages,
                  generate_kwargs={"max_new_tokens": 20},
                  return_full_text=False)
    
    candidate_caption = output[0]['generated_text']
    reference_caption = reference_list[i]
    print("Candidate caption: {}".format(candidate_caption))
    print("Reference caption: {}".format(reference_caption))

    judge_prompt = judge_score_template.format(ref=reference_caption, cand=candidate_caption)

    messages2 = [
        {"role": "user",
        "content": [{"type": "image"},
                    {"type": "text", "text": judge_prompt}]
        },
    ]

    output2 = pipe2(images=entry['image'],
                  text=messages2,
                  generate_kwargs={"max_new_tokens": 10},
                  return_full_text=False)
    judge_score = output2[0]['generated_text']
    print("LMM Judge: {}".format(judge_score))
    
    messages3 = [
        {"role": "user",
        "content": [{"type": "image"},
                    {"type": "text", "text": judge_prompt}]
        },
        {"role": "assistant",
         "content": [{"type": "text", "text": judge_score + "/ 1.0"}]},
        {"role": "user",
         "content": [{"type": "text", "text": judge_followup_prompt}]},
    ]

    output3 = pipe3(images=entry['image'],
                  text=messages3,
                  generate_kwargs={"max_new_tokens": 50},
                  return_full_text=False)
    
    judge_reason = output3[0]['generated_text']
    print("Judge Reason: {}".format(judge_reason))


