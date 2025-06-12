# https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf
# https://huggingface.co/datasets/MiXaiLL76/TextOCR_OCR

from transformers import pipeline
from datasets import load_dataset
import evaluate
exact_match_metric = evaluate.load("exact_match")

ds = load_dataset("MiXaiLL76/TextOCR_OCR", split='train')
shuffled_ds = ds.shuffle()

num_samples_to_process = 10
ds_subset = shuffled_ds.select(range(num_samples_to_process))
print(ds_subset)
prompt = "You are an optical character recognition tool. Your task is process the text given in the attached image."\
"Analyze the image and respond with the text shown. Do not provide any explanation or introductory text, or anything other than what is shown."

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
