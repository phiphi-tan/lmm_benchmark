
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B", torch_dtype="auto", device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-Omni-3B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {"role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Who are you?"}]},
]

# set use audio in video
# USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
inputs = inputs.to(model.device).to(model.dtype)
print(inputs)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs)
print(text_ids)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)

