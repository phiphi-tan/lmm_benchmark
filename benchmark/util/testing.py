
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from datasets import load_dataset
import torch
# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B", torch_dtype=torch.bfloat16, device_map="auto")

# # We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-Omni-3B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
model.disable_talker()

ds = load_dataset("vikhyatk/tallyqa-test", split="test")

print("Original Dataset: {}".format(ds))
input_dataset = ds.select(range(1))
print("Sampled Dataset: {}".format(input_dataset))

image_list = input_dataset['image']
qa_list = input_dataset['qa']
question_list = [q[0]["question"] for q in qa_list]
answer_list = [q[0]["answer"] for q in qa_list]

print(image_list)
print(question_list)
print(answer_list)

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are an object recognition tool"}
        ],
    },
    {"role": "user",
            "content": [{"type": "image", "image": image_list[0]},
                        {"type": "text", "text": question_list[0]}]},
]


# set use audio in video
# USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
inputs = inputs.to(model.device).to(model.dtype)
input_length = inputs.input_ids.shape[1]


# Inference: Generation of the output text and audio
print('generating output')

text_ids = model.generate(**inputs, return_audio=False)
text = processor.batch_decode(text_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)

