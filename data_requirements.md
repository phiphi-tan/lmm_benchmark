## Data Requirements
To remain relevant for the defence domain, this benchmark is specifically curated to meet the data requirements at 4 initial basic sub-tasks that are essential for many downstream tasks.

These sub-tasks may also be split into further nested sub-tasks where necessary for better clarity and to ensure the breadth of the dataset fulfills the requirements.

An elaboration of each sub-task (and any nested sub-tasks) are below.

---

### 1. Object Detection
Object detection refers to the task of locating objects in provided inputs (in this case, image and possibly video). In this particular case, there are certain categories of objects that we wish to identify.

#### Classes
The dataset should be comprehensive enough to cover different types of objects; a non-exhaustive list is as follows:
- Military assets
- Military / political personnel
- Weapons / tools
- Logos (Overlap with OCR)

#### Range
The dataset should contain a range of different conditions, differing with location (indoor / outdoor) and contextual scenarios (urban / rural)

The dataset should also contain variation in perspectives, such as top-down (e.g. aerial views / maps), side angles, and oblique perspectives (e.g. ultrawide / fisheye lens)

#### Annotation
To ensure proper identification of the object, outputs should be annotated with bounding boxes with clases labels. Segmentations can also be provided if needed for more precise object localisation.

#### Sub-categories
> For now, there are no sub-categories for Object Detection.

### 2. Counting

Counting refers to the task of identifying and counting specific objects within the given inputs.

> Note: While traditionally object counting may be seen as a subset of [Object Detection](#1-object-detection), there are certain techniques that models may use to create more accurate answers than simply aggregating object detection counts.

#### Classes
The dataset should encompass repetitive patterns, such as crowds, carparks and vehicle parades, and be able to identify and count objects within them.

#### Range
The dataset should contain variations of object sizes and viewpoints (e.g. top-down, side perspectives)

#### Annotation
The model should provide an exact count of objects within the provided input. Bounding boxes / region-based annotations / target points can also be provided if available.

#### Sub-categories
As seen from the [TallyQA study](https://arxiv.org/abs/1810.12440), counting problems can be mainly categories into two different sub-categories

1. Simple counting: *(i.e. "How many dogs are there?")*
2. Complex counting: *(i.e. "How many dogs are eating?")*

Our dataset should comprise of enough data samples to cover both of these sub-categories, which also ensures the the model is robust enough to handle more complex counting requests (a large proportion of datasets revolve more around simpler counting questions).

### 3. OCR (Optical Character Recognition)
Optical Character Recognition refers to the task of converting images of text into machine-readable text format. Essentially, OCR's usage involves tasks like converting scanned documents into the computer text.

#### Classes
The dataset should be able to accept various text-based inputs; a non-exhaustive list is as follows:
- Scanned documents / presentation slides
- Photos of printed text
- Handwriting samples
- Signboards

#### Range
The dataset should contain variations of machine-printed / hand-written text.

The dataset should also span across different text orientations, font sizes and handwriting types.

#### Annotation
The model should provide transcriptions of the text in the input provided, as well as bounding boxes around text regions transcribed. There should also be selective multi-language type output as well.

#### Sub-categories
Possible sub-categories can include differentiating between machine-printed and handwritten text. 

### 4. Image Captioning
Image captioning refers to the task of describing the content of a given image.

> Image captioning can also be seen as a extension of object detection, as most models utilise some form of object detection within their architecture. The dataset will hence see some overlaps with [Object Detection](#1-object-detection)

#### Classes
The dataset should be able to take in any form of input image and provide an appropriate caption for it.

#### Range
The dataset should contain a wide range of input scenes (e.g. daily life, nature, events, specific tasks, nouns), maintaining diversity across geography, culture and context.

#### Annotation
The model should output descriptive captions that effectively summarise the content of the image, with multiple different captions per image to provide variability and richness.

#### Sub-categories
> Due to the wide input range, there are different ways to categorise the inputs (e.g. nouns vs verbs / input environment)
