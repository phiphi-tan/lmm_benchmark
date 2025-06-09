# https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf
# https://huggingface.co/datasets/mehul7/captioned_military_aircraft

from transformers import pipeline
from datasets import load_dataset
import evaluate
exact_match_metric = evaluate.load("exact_match")

ds = load_dataset("mehul7/captioned_military_aircraft", split="train")
shuffled_ds = ds.shuffle()

num_samples_to_process = 10
ds_subset = shuffled_ds.select(range(num_samples_to_process))
print(ds_subset)
prompt = "You are a military aircraft recognition tool. Your task is identify the aircraft shown in the attached image."\
"Analyze the image and respond with the name of the exact model of the aircraft shown. Do not provide any explanation or introductory text, or anything other than the name of the aircraft in the scene."

pipe = pipeline("image-text-to-text", model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
prediction_list = [None for _ in range(num_samples_to_process)]
reference_list = ds_subset['text']
for i, entry in enumerate(ds_subset):
    messages = [
        {"role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    output = pipe(images=entry['image'],
                  text=messages,
                  generate_kwargs={"max_new_tokens": 20},
                  return_full_text=False)
    prediction_list[i] = output[0]['generated_text']
    
print("Predicted values: {}".format(prediction_list))
print("Actual values: {}".format(reference_list))

results = exact_match_metric.compute(references=reference_list, predictions=prediction_list)
print("Exact Match Rate: {}".format(round(results["exact_match"], 2)))

# Predicted values: ['The aircraft shown in the image is the RQ-4.', 'The aircraft is a C-130', 'The aircraft shown in the image is the F-16 Fighting Falcon.', 'The aircraft shown in the image is the B-17 Flying Fortress.', 'The aircraft shown in the image is the DHC-1', 'The aircraft shown in the image is the C-130 Hercules.', 'The aircraft shown in the image is the F-16 Fighting Falcon.', 'The aircraft is a F-16 Fighting Falcon', 'The aircraft shown in the image is the F-16 Fighting Falcon.', 'The aircraft shown in the image is the F/A-18 Hornet.']
# Actual values: ['a large propeller mq9 plane sitting on top of an airport tarmac', 'a v22 plane flying in the sky over a field', 'a f15 fighter jet flying over a lush green field', 'a v22 plane sitting on top of an airport tarmac', 'us2 airplanes parked on the tarmac of an airport', 'an c130 airplane that is flying in the sky', 'a dog standing on top of an airport tarmac with a f35', 'a f16 fighter jet flying through a cloudy sky', 'f16 airplanes flying in formation in the sky', 'a man flying through the air while holding an v22 airplane']
# Match Rate (manual): 0.2