# https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf
# https://huggingface.co/datasets/TrainingDataPro/crowd-counting-dataset

from transformers import pipeline
from datasets import load_dataset

ds = load_dataset("TrainingDataPro/crowd-counting-dataset", split='train')
shuffled_ds = ds.shuffle()

num_samples_to_process = 5
ds_subset = shuffled_ds.select(range(num_samples_to_process))
print(ds_subset)
prompt = "You are an image analysis tool. Your task is to estimate the crowd size in the provided image."\
"Analyze the image and respond with a single number. Do not provide any explanation or introductory text, or any additional symbols or units of measurements."


pipe = pipeline("image-text-to-text", model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
output_list = [None for _ in range(num_samples_to_process)]
print(output_list)

for i, entry in enumerate(ds_subset):
    messages = [
        {"role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    output = pipe(images=entry['image'],
                  text=messages,
                  generate_kwargs={"max_new_tokens": 20},
                  return_full_text=False)
    output_list[i] = output[0]['generated_text']
    
print("Predicted values: {}".format(output_list))
print("Actual values: {}".format(ds_subset['label']))