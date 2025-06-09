# https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf
# https://huggingface.co/datasets/Naveengo/flickr8k

from transformers import pipeline
from datasets import load_dataset
import evaluate
# google bleu used as better for single-sentence evaluations
google_bleu = evaluate.load("google_bleu")

ds = load_dataset("Naveengo/flickr8k", split='train')
shuffled_ds = ds.shuffle()

num_samples_to_process = 5
ds_subset = shuffled_ds.select(range(num_samples_to_process))
print(ds_subset)
prompt = "You are an image caption tool. Your task is generate captions for a given input image."\
"Analyze the image and respond with an appropriate caption. Do not provide any explanation or introductory text, or anything other than what is shown."

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

for i in range(num_samples_to_process):
    print("Sample {}".format(i))
    print("Generated Caption: {}".format(prediction_list[i]))
    print("Reference Caption: {}".format(reference_list[i]))

results = google_bleu.compute(predictions=prediction_list, references=reference_list)
print("Google BLEU Score: {}".format(round(results["google_bleu"], 2)))

# Sample 0
# Generated Caption: a person standing on a rock looking at the stars
# Reference Caption: A person in the distance hikes among hoodoos with stars visible in the sky .
# Sample 1
# Generated Caption: a group of people standing in the grass
# Reference Caption: A group stands in the distance while the sky casts interesting light on some clouds .
# Sample 2
# Generated Caption: people waiting for a train at the subway station
# Reference Caption: "A crowd waits outside a subway train   ready to board ."
# Sample 3
# Generated Caption: a boy jumping into a pool
# Reference Caption: A boy is diving through the air into a swimming pool .
# Sample 4
# Generated Caption: a deer walking through the snow with a group of wild turkeys
# Reference Caption: a deer and several turkeys together in the snow
# Sample 5
# Generated Caption: a surfer in the air after a wipeout
# Reference Caption: a lone surfboarder jumping a wave on a white surfboard .
# Sample 6
# Generated Caption: a person carrying a bucket over their head
# Reference Caption: A boy wearing brown shorts pouring water over his head .
# Sample 7
# Generated Caption: a girl in a pink jacket with her arms outstretched
# Reference Caption: A girl in a pink coat plays in the leaves .
# Sample 8
# Generated Caption: a man walking down the sidewalk with a bag
# Reference Caption: "A man carries two grocery bags   a sign of a funny man is by the pole near him ."
# Sample 9
# Generated Caption: a dog running in the water
# Reference Caption: A black and brown dog is running out of the surf .
# Google BLEU Score: 0.09
