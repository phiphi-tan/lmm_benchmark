from .benchmark_tools import eval_results
from .bounding_boxes import eval_bbox
from word2number import w2n
import ast
from datasets import load_dataset, Dataset

# dataset_path = "pathikg/drone-detection-dataset"
# dataset_split = "test"
# sample_size = 1
# ds = load_dataset(dataset_path, split=dataset_split)

# print("Original Dataset: {}".format(ds))
# # filter for only single-drone detection
# ds = ds.filter(lambda row: len(row['objects']['category']) == 1)
# print("Filtered Dataset: {}".format(ds))
# shuffled_ds = ds.shuffle(seed=sample_size) # for random selection
# input_dataset = shuffled_ds.select(range(sample_size))

# image_list = input_dataset['image'] # get list of images
# ref_data_list = input_dataset['objects'] # get list of answers
# ref_data_list = [r['bbox'] for r in ref_data_list]

# print(input_dataset)
# print(image_list)
# print(ref_data_list)

ref_list = ['1', '1', '1', '1', '1', '4', '1', '1', '2', '2', '1', '1', '1', '6', '2', '1', '3', '1', '2', '3', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '2', '1', '1', '1', '1', '1', '1', '3', '1', '2', '1', '2', '2', '1', '2', '1', '1', '2', '2', '2', '1', '1', '1', '1', '1', '1', '1', '1', '4', '2', '2', '2']

pred_list = ['1', '1', 'one', 'One', '1', '4', '1', '1', 'one', 'two', '1', 'One', 'one', '7', '2', '1', 'three', 'one', '2', 'three', 'two', 'One', 'One', '1', 'one', '1', '1', 'One', '1', '1', '1', '1', 'two', '1', '1', 'One', '1', '1', '1', 'two', '1', '2', '1', '2', '2', 'one', 'two', 'One', '2', '1', 'two', 'two', 'One', 'one', 'one', 'One', '1', '1', 'one', '1', '1', '2', 'two', '2']


# print(eval_bbox(ref_list, pred_list, normalise=True, img_list=image_list))
# ref_list = [s.upper() for s in ref_list]
# pred_list = [s.upper() for s in pred_list]

pred_list = [str(w2n.word_to_num(i)) for i in pred_list]

print(eval_results(ref_list, pred_list, 'exact_match'))
# print(eval_results(ref_list, pred_list, 'bbox_iou'))

# score_list = [0.0, 0.0, 0.61, 0.0, 0.24, 0.48, 0.03, 0.0, 0.7, 0.54, 0.82, 0.85, 0.81, 0.89, 0.51, 0.0, 0.0, 0.61, 0.31, 0.0, 0.63, 0.84, 0.67, 0.58, 0.0, 0.78, 0.06, 0.54, 0.67, 0.75, 0.51, 0.57, 0.55, 0.34, 0.88, 0.0, 0.15, 0.43, 0.54, 0.0, 0.81, 0.62, 0.1, 0.0, 0.82, 0.65, 0.74, 0.4, 0.0, 0.38, 0.75, 0.59, 0.49, 0.0, 0.0, 0.84, 0.0, 0.0, 0.0, 0.89, 0.78, 0.0, 0.75, 0.77] 
# avg = sum(score_list) / len(score_list)
# print(round(avg, 2))

# pred = ['[62,291,170,378]', '[16,142,37,156]', '[378,196,1987,1254]', '[95,487,620,761]', '[38,62,190,157]', '[10,5,240,130]', '[72,49,218,99]', '[73,29,145,180]', '[38,56,432,307]', '[10,10,200,150]', '[139,156,2897,1380]', '[10,10]', '[28,23,301,274]', '[10,10], [200,150]', '[10,10]', '[514,60,839,247]', '[194,206,273,245]', '[10,10], [200,150]', '[73,40,859,592]', '[135,248,176,335]', '[235,41,1030,667]', '[10,10][200,150]', '[253,148,1740,946]', '[1,124,336,504]', '[90,302,145,387]', '[0,34,438,345]', '[45,60,132,107]', '[10,10]', '[0,123,584,504]', '[54,36,476,291]', '[10,10,200,150]', '[127,14,439,360]', '[30,85,486,329]', '[453,168,1057,552]', '[10,10][200,150]', '[327,410,420,465]', '[165,200,379,471]', '[30,92,468,261]', '[61,63,312,315]', '[287,79,473,265]', '[49,83,337,295]', '[10,10]', '[145,83,346,209]', '[137,42,230,98]', '[10,10,200,180]', '[10,10]', '[82,30,665,261]', '[90,148,643,427]', '[154,78,209,263]', '[10,10,120,120]', '[37,0,219,86]', '[10,10]', '[10,10][20,20]', '[268,3,437,501]', '[798,0,1235,924]', '[10,10][200,200]', '[36,67,102,125]', '[176,120,285,269]', '[1,204,178,413]', '[25,17,483,289]', '[10,10,200,150]', '[276,78,504,319]', '[94,0,504,364]', '[10,10][200,150]']

# new_pred = []

# for p in pred:
#     try:
#         new_pred.append(ast.literal_eval(p))
#     except:
#         new_pred.append(p)

# print(pred)
# print(new_pred)
