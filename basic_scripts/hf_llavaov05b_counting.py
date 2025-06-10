# https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf
# https://huggingface.co/datasets/nimapourjafar/mm_tallyqa

from transformers import pipeline
from datasets import load_dataset
from PIL import Image
import io

ds = load_dataset("nimapourjafar/mm_tallyqa", split='train')
shuffled_ds = ds.shuffle()

num_samples_to_process = 3
ds_subset = shuffled_ds.select(range(num_samples_to_process))
print(ds_subset)
system_prompt = "You are an object counting tool. Your task is to estimate the number of objects in the provided image. "\
"Analyze the image and respond with a single number. Do not provide any explanation or introductory text."

data_list = ds_subset['data']
# only use the first QA for each image
question_data_list = [x[1] for x in data_list]
question_data_list = [q['data'] for q in question_data_list]
print("Question List: {}".format(question_data_list))

answer_data_list = [x[2] for x in data_list]
answer_data_list = [q['data'] for q in answer_data_list]
print("Answer List: {}".format(answer_data_list))

reference_list = []

pipe = pipeline("image-text-to-text", model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
prediction_list = [None for _ in range(num_samples_to_process)]
reference_list = answer_data_list

for i, entry in enumerate(ds_subset):
    # dataset image is in bytes format, need to convert to PIL
    img_bytes = entry['images'][0]['bytes']
    img = Image.open(io.BytesIO(img_bytes))
    # print(img)
    img.show()
    # display(img) # for Google Colab

    user_prompt = question_data_list[i]
    messages = [
        {"role": "system",
        "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
    ]
    output = pipe(images=img,
                  text=messages,
                  generate_kwargs={"max_new_tokens": 20},
                  return_full_text=False)

    prediction = output[0]['generated_text']

    print("Question: {}".format(user_prompt))
    print("Predicted Answer: {}".format(prediction))
    print("Actual Answer: {}".format(reference_list[i]))

    prediction_list[i] = prediction

    
# for i in range(num_samples_to_process):
#     print("Sample {}".format(i))
#     print("Generated Count: {}".format(prediction_list[i]))
#     print("Reference Count: {}".format(reference_list[i]))