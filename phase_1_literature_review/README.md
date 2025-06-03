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
> Honestly, just a lot of these QA Frameworks for now
- DocVQA (Document Visual Question Answering)
    - Open-ended questions from provided document with ground-truth annotations
    - Link: https://www.docvqa.org/datasets

- InfographicVQA
    - 30,000 questions with 5,000 images (collected from the internet) and manually annotated
    - Link: https://www.docvqa.org/datasets/infographicvqa

- ChartQA
    - Large-scale benchmark covering 9,600 human-written questions as well as 23,100 questions generated from human-written chart summaries
    - Link: https://arxiv.org/abs/2203.10244

- IAM Handwriting
    - 13,353 images of handwritten lines of text created by 657 writers
    - Link: https://paperswithcode.com/dataset/iam

- Handwriting Recognition (Names)
    - 206,799 first names, 207,024 surnames
    - Link: https://www.kaggle.com/datasets/ssarkar445/handwriting-recognitionocr


### 3. Counting Data
> Note: Given that counting data can be seen as a subset of object detection, many of the datasets used in [Object Detection](#1-object-detection-data) can also be used for this sub-task.

> The datasets listed below are more specific for this sub-task.

- VQA (Visual Question Answering)
    - Open-ended questions regarding provided images with multiple questions + ground-truth answers per image
    - Two dataset versions available (v1 & v2)
    - Link: https://visualqa.org/

- **CounterCurate**
    - > Interesting read
    - Data augmentation through image generation in order to improve objecting counting accuracy
    - Introduces generated contrastive inputs into the dataset to help with training
    - Improves performance across contrastive (CLIP) and generative (LLaVa) models
    - Link: https://countercurate.github.io/
    - Link: https://github.com/HanSolo9682/CounterCurate

- **PointQA** 
    - > Interesting read
    - > The LookTwice benchmark is used in the above CounterCurate paper for object counting
    - Includes spatial point of reference alongside the image input, leading to higher accuracy
    - Three different datasets
        - Local: Local regions ("What colour is this train")
        - LookTwice (relevant): Local regions in broader context of image ("how many of *this* animal are there?)
        - General: ("is this part of the computer touching the carpet?")
    - Link: https://paperswithcode.com/dataset/pointqa
    - Link: https://arxiv.org/pdf/2011.13681v4

- **TallyQA**
    - > Interesting read
    - Huge open-ended counting dataset, split into simple and complex (human-vetted), over 250,000 questions and 150,000 images
    - Differentiates between non-complex ("how many dogs are there") vs complex ("how many dogs are eating") counting questions (linguistics)
    - Introduces a new algorithm - Relational counting network which compares relationships between objects
    - Link: https://arxiv.org/abs/1810.12440
    - Link: 

- HowManyQA
    - > Used in the above TallyQA study
    - Link: https://paperswithcode.com/dataset/howmany-qa

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

> Similarly to [Counting Data](#3-counting-data), some portions of this sub-task intersect with [Object Detection](#1-object-detection-data). I will briefly mention these intersections, but leave the elaboration in the previous sub-task.

- NOVAC (uNconstrained Open Vocabulary Image Classifier)
    - > Interesting read
    - Builds upon current data classification by introducing a decoder for CLIP (contrastive language image pre-training) text embeddings
    - Removes the need for any initial classifiers, and will output open-ended captions based on input image
    - Link: https://paperswithcode.com/paper/unconstrained-open-vocabulary-image
    - Link: https://arxiv.org/pdf/2407.11211v4

- MS COCO Captions
    - > Mentioned above in Object Detection.
    - Dataset contains 5 captions per image
    - Link: https://arxiv.org/abs/1504.00325

- Polos
    - > Interesting read
    - Supervised automatic evaluation metric for image captioning models
    - Also introduces M2LHF (Multimodal Metric Learning from Human Feedback), a novel framework for metric development using human feedback
    - Link: https://paperswithcode.com/paper/polos-multimodal-metric-learning-from-human

- Polaris Dataset
    - > Dataset used in the above Polos study
    - "Comprises 131K human judgements collected from 550 evaluators. Our dataset further distinguishes itself from existing datasets in terms of the inclusion of diverse captions, which are collected from humans and generated from ten image captioning models, including modern models"
    - Link: https://huggingface.co/datasets/yuwd/Polaris

- Satellite Image
    - 10,000 Satellite images accompanied by 5 different captions (annoted by 5 different annotators)
    - Collected from Google Earth, Baidu Map, MapABC, Tianditu
    - Link: https://www.kaggle.com/datasets/tomtillo/satellite-image-caption-generation
    - Link: https://arxiv.org/pdf/1712.07835

- Google's Quality Estimation for Image Captions
    - Quality estimation for machine-generated image captions using crowdsourced ratinga 
    - Link: https://github.com/google-research-datasets/Image-Caption-Quality-Dataset

## Processing
More information on this will be in `phase_3_dataset_handling/README.md`.

To note: Some literature review regarding data filtering and data augmentation may be needed as well.

### Data Filtering Approaches

### Data Augmentation Approaches




