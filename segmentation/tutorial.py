import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from ..benchmark.util.bounding_boxes import draw_bboxes

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
image.show()

text = "a cat. a remote control."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

score_list = results['scores']
bbox_list = results['boxes']
text_labels_list = results['text_labels']
labels_list = results['labels'] # not sure what the diff is for this

num_objects_detected = len(score_list)
colours = ['red', 'blue', 'green', 'yellow', 'white', 'grey'] # hardcoded colour list for now
for i in range(num_objects_detected):
    image = draw_bboxes(image, )
    image.show()
    





print(results)
