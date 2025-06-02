# Literature Review
> Literature review on multimodal data curation for evaluation benchmarks.

The goal for this document is to list down relevant literature findings for the assigned problem (data curation for evaluation benchmarks)

## Summary
There are 4 main sub-tasks essential for many downstream tasks, and hence curating data for these subtasks may have different research / requirements associated with it. There is also a section for general papers, which may apply to more than one sub-task (or only closely related but still included for documentation)

### General Papers

1. Google's Gemini Model
    - Google's paper on their Gemini model
    - Provides different referenced benchmarks for text + image + video + audio
    - References natural image understanding -- may be useful for captioning data
    - Link: https://arxiv.org/abs/2312.11805

- LMMsEval
    - "Unified and standardized multimodal benchmark suite"
    - Link: https://arxiv.org/abs/2407.12772v1
    - Link: https://github.com/EvolvingLMMs-Lab/lmms-eval

- MMStar
    - "MMStar, an elite vision-indispensable multi-modal benchmark comprising 1,500 samples meticulously selected by humans. MMStar benchmarks 6 core capabilities and 18 detailed axes, aiming to evaluate LVLMs’ multi-modal capacities with carefully balanced and purified samples."
    - Link: https://mmstar-benchmark.github.io/




### 1. Object Detection Data
> Difficult to find military asset data
- RF100 (Roboflow 100)
    - The datasets are splitted in 7 categories: Aerial, Videogames, Microscopic, Underwater, Documents, Electromagnetic and Real World
    - Link: https://github.com/roboflow/roboflow-100-benchmark

- MS COCO (Microsoft Common Objects in Context)
    - Large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images
    - Link: https://paperswithcode.com/dataset/coco

- SkyFusion
    - Vehicle detection from satellite images (aerial) -- "tiny object detection"
    - Link: https://www.kaggle.com/datasets/kailaspsudheer/tiny-object-detection

- Military Aircraft Recognition
    - 81 different military aircraft types
    - Link https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset


### 2. OCR (Optical Character Recognition) Data:
> Honestly, just a lot of QA Frameworks for now
- DocVQA (Document Visual Question Answering)
    - Open-ended questions from provided document with ground-truth annotations
    - Link: https://www.docvqa.org/datasets

- InfographicVQA
    - 30,000 questions with 5,000 images (collected from the internet) and manually annotated
    - Link: https://www.docvqa.org/datasets/infographicvqa

- ChartQA
    - Large-scale benchmark covering 9,600 human-written questions as well as 23,100 questions generated from human-written chart summaries
    - Link: https://arxiv.org/abs/2203.10244


### 3. Counting Data
> Note: Given that counting data can be seen as a subset of object detection, many of the datasets used in [Object Detection](#1-object-detection-data) can also be used for this sub-task

- VQA (Visual Question Answering)
    - Open-ended questions regarding provided images with multiple questions + ground-truth answers per image
    - Two dataset versions available (v1 & v2)
    - Link: https://visualqa.org/

- CounterCurate
    - > Interesting read
    - Data augmentation through image generation in order to improve objecting counting accuracy
    - Link: https://countercurate.github.io/
    - Link: https://github.com/HanSolo9682/CounterCurate

- TallyQA
    - > Interesting read
    - Huge open-ended counting dataset, split into simple and complex (human-vetted), over 250,000 questions and 150,000 images
    - Differentiates between non-complex ("how many dogs are there") vs complex ("how many dogs are eating") counting questions (linguistics)
    - Introduces a new algorithm - Relational counting network which compares relationships between objects

- COWC (Cars Overhead With Context)
    - 32,716 unique annotated cars. 58,247 unique negative examples.
    - Link: https://gdo152.llnl.gov/cowc/
    - Link: https://paperswithcode.com/dataset/cowc

- FSC147
    - 147 object categories over 6000 images
    - Annotated with a dot at image approximate center
    - Link: https://paperswithcode.com/dataset/fsc147

- Crowd Counting (Video)
    -  60,000 pedestrians were labelled in 2000 video frames (from a single public webcam)
    - Annotated via head labels
    - Link: https://www.kaggle.com/datasets/fmena14/crowd-counting


### 4.  Captioning Data
> Due to the open-ended nature of this question, benchmarking this sub-task may be different from the other sub-tasks.

- Satellite Image
    - 10,000 Satellite images accompanied by 5 different captions (annoted by 5 different annotators)
    - Link: https://www.kaggle.com/datasets/tomtillo/satellite-image-caption-generation

- Google's Quality Estimation for Image Captions
    - > May not be directly related
    - Crowdsourced ratings for machine-generated image captions
    - Link: https://github.com/google-research-datasets/Image-Caption-Quality-Dataset

## Post-processing
More information on this will be in `phase_3_dataset_handling/README.md`.

Basically, some literature review regarding data filtering and data augmentation may be needed as well.

### Data Filtering Approaches
1. Manual Filtering


### Data Augmentation Approaches




