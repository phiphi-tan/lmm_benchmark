# Benchmark Results
This details the initial results found from the testing of different models

> Note: The models used are of a relatively small size

## Object Counting
### Dataset
The dataset comprises of images from TallyQA

### Evaluation
The evaluation metric used is exact match, and will directly compare the output from the model against the reference answers given in the dataset

### Results

Initial testing on a small batch of 10 had mixed results. In general the model trained with higher parameters (i.e. Qwen2-2B) performed better. However, many runs had the smaller SmolVLM-256M model performing better than Llava-OV-0.5B

Results of the most recent runs are as below:

Batch size of 10
```
Qwen/Qwen2-VL-2B-Instruct: 0.8
llava-hf/llava-onevision-qwen2-0.5b-ov-hf: 0.6
HuggingFaceTB/SmolVLM-256M-Instruct: 0.8
```

Batch size of 50:
```
Qwen/Qwen2-VL-2B-Instruct: 0.72
llava-hf/llava-onevision-qwen2-0.5b-ov-hf: 0.68
HuggingFaceTB/SmolVLM-256M-Instruct: 0.82
```

Batch size of 100:
```
Qwen/Qwen2-VL-2B-Instruct: 0.74
llava-hf/llava-onevision-qwen2-0.5b-ov-hf: 0.7
HuggingFaceTB/SmolVLM-256M-Instruct: 0.79
```

## Optical Character Recognition
### Dataset
The dataset comprises of simple straightforward images of text, taken from TextOCR

### Evaluation
The evaluation metric used is exact match, and will directly compare the output from the model against the reference answers given in the dataset.

A non-exact (case insensitive) metric was manually calculated for the smaller batch of 10

### Results
> I had to manually edit the output of the SmolVLM-256M model to remove any punctuation and leading/trailing whitespaces. Without which the performance would be 0.

However, results are inconsistent as models occasionally change the capitalisation of the text, most specifically the Qwen2-2B model. I have manually included a case-insensitive version of a batch size of 10 to show the drastic increase in evaluated performance.

Results of the most recent runs are as below:

Batch size of 10
```
Qwen/Qwen2-VL-2B-Instruct: 0.3
Qwen/Qwen2-VL-2B-Instruct (case-insensitive): 0.9
llava-hf/llava-onevision-qwen2-0.5b-ov-hf: 0.6
llava-hf/llava-onevision-qwen2-0.5b-ov-hf (case-insensitive): 0.7
HuggingFaceTB/SmolVLM-256M-Instruct: 0.4

```

Batch size of 50:
```
Qwen/Qwen2-VL-2B-Instruct: 0.2
llava-hf/llava-onevision-qwen2-0.5b-ov-hf: 0.4
HuggingFaceTB/SmolVLM-256M-Instruct: 0.32
```

Batch size of 100:
```

```

## Object Detection (Military Aircraft Classification)
### Dataset
The dataset comprises of aircraft images, each with a classification label of either civilian or military.

### Results
Surprisingly, the Qwen2-2B model performed the worst despite the larger trained instruction set, constantly outputting a single value of `1 (military)` for any image.

Results of the most recent runs are as below:

Batch size of 10
```
Qwen/Qwen2-VL-2B-Instruct: 0.4
llava-hf/llava-onevision-qwen2-0.5b-ov-hf: 0.2 
HuggingFaceTB/SmolVLM-256M-Instruct: 0.5
```

Batch size of 50:
```
Qwen/Qwen2-VL-2B-Instruct: 0.2
llava-hf/llava-onevision-qwen2-0.5b-ov-hf: 0.4
HuggingFaceTB/SmolVLM-256M-Instruct: 0.32
```

## Object Detection (Pedestrians in Fog)
### Dataset
The dataset comprises of images of pedestrians

### Results


```

```




## Conclusion
