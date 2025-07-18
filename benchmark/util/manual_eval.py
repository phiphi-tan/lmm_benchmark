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

# ref_list =  ['Performance', 'Sport', '...period.', 'GTOR®', '12', 'CAOL', 'DISTILLERY', 'G-ATCO', 'OUR', 'NEIGHBORS,', 'THE', 'FRIENDS', 'Muhomah', 'friends', '97215', 'NORTHWEST', 'THE', 'WHITEHEADED', 'ÚJSÁG', "WAGNER'ÚR?", 'PARTI', 'Chrstphre', 'Campbell', '903', 'West', 'Spofford', 'Unit', 'Seven', 'Spokane', 'Washington', '99205', '·DEPART', 'PARTI', 'DAN', 'DES', 'USE', 'KA', '003-', 'OV', 'FORTY', 'NINERS', 'S', 'conesa', 'conesa', 'Fanta', 'pper.', 'Pepper.', 'Imperial', 'Dr', 'Pepper', 'Dr', 'Pep', 'Pe', 'D', "mickey's", 'mainsall', 'Mainsall', 'WALT', 'DISNEY', 'STUDIOS', 'PgDn', 'PgUp', 'Home', 'Microsoft']

# preds_list = [
#     [' Performance ', ' Sport ', ' Period. ', ' Ct ', ' 12 ', ' CAL ', ' Whisky distillery ', ' G-ATCD ', ' Our ', ' NEIGHBORS ', ' The text in the image is "THE". ', ' Friends ', ' Muhom ', ' The image is too blurry to read the text. ', ' 97215 ', ' Northwest ', ' The text shown in the image is "THE". ', ' WHITEHEAD ', ' Uusag, uusag, uusag, uusag, uusag, uusag, uusag, uusag, uusag, uusag, uusag, uusag, uus', ' Wagener? ', ' The text shown in the image is "Bottleville, Stix 666000". ', ' Christophre ', ' Campbell ', ' 903 ', ' West ', ' Spofford ', ' Unit ', ' Seven ', ' Spokane ', ' Washington ', ' 99205 ', ' Department ', ' The image is too blurry to read the text. ', ' Dan ', ' DES ', ' Use ', ' KA ', ' 000 000 000 000 000 000 000 000 000 000 000 000 0', ' O ', ' FORTH ', ' NINERS ', ' The image is too blurry to read the text. ', ' Conesa ', ' Conexa ', ' Fanta ', ' Pepsi ', ' Dr. Pepper ', ' The text shown in the image is "penal". ', ' Dr pepper ', ' Dr Pepper ', ' The text shown in the image is "Pepsi". ', ' Pepsi ', ' The image is too blurry to read the text. ', ' The image is too blurry to read the text. ', " Mickey's ", ' MANSA ', ' The image is too blurry to read the text. ', ' WALT ', ' Disney ', ' Studio ', ' PgDn ', ' pgup ', ' Home ', ' Microsoft '],
# ]
# # print(eval_bbox(ref_list, pred_list, normalise=True, img_list=image_list))
# ref_list = [s.upper() for s in ref_list]
# preds_list = [[s.strip().upper() for s in pred_list] for pred_list in preds_list]

# # pred_list = [str(w2n.word_to_num(i)) for i in pred_list]
# for pred_list in preds_list:
#     print(eval_results(ref_list, pred_list, 'exact_match'))

# print(eval_results(ref_list, pred_list, 'bbox_iou'))

score_list =   [0.8297, 0.3292, 0.96, 0.5667, 0.8018, 0.6016, 0.6664, 0.3743, 0.9349, 0.5499, 0.8836, 0.8552, 0.9662, 0.884, 0.5229, 0.9531, 0.4991, 0.5659, 0.3917, 0.7253, 0.9834, 0.8366, 0.9802, 0.9517, 0.4727, 0.8384, 0.3747, 0.5276, 0.9188, 0.9559, 0.5298, 0.9588, 0.9274, 0.8542, 0.8858, 0.5305, 0.7787, 0.9713, 0.9606, 0.929, 0.5037, 0.5748, 0.9829, 0.816, 0.791, 0.6379, 0.9205, 0.9565, 0.8212, 0.38, 0.8946, 0.59, 0.5816, 0.7866, 0.7259, 0.8518, 0.7687, 0.891, 0.7808, 0.9334, 0.7659, 0.792, 0.9585, 0.7669]
avg = sum(score_list) / len(score_list)
print(round(avg, 3))

# pred = ['[62,291,170,378]', '[16,142,37,156]', '[378,196,1987,1254]', '[95,487,620,761]', '[38,62,190,157]', '[10,5,240,130]', '[72,49,218,99]', '[73,29,145,180]', '[38,56,432,307]', '[10,10,200,150]', '[139,156,2897,1380]', '[10,10]', '[28,23,301,274]', '[10,10], [200,150]', '[10,10]', '[514,60,839,247]', '[194,206,273,245]', '[10,10], [200,150]', '[73,40,859,592]', '[135,248,176,335]', '[235,41,1030,667]', '[10,10][200,150]', '[253,148,1740,946]', '[1,124,336,504]', '[90,302,145,387]', '[0,34,438,345]', '[45,60,132,107]', '[10,10]', '[0,123,584,504]', '[54,36,476,291]', '[10,10,200,150]', '[127,14,439,360]', '[30,85,486,329]', '[453,168,1057,552]', '[10,10][200,150]', '[327,410,420,465]', '[165,200,379,471]', '[30,92,468,261]', '[61,63,312,315]', '[287,79,473,265]', '[49,83,337,295]', '[10,10]', '[145,83,346,209]', '[137,42,230,98]', '[10,10,200,180]', '[10,10]', '[82,30,665,261]', '[90,148,643,427]', '[154,78,209,263]', '[10,10,120,120]', '[37,0,219,86]', '[10,10]', '[10,10][20,20]', '[268,3,437,501]', '[798,0,1235,924]', '[10,10][200,200]', '[36,67,102,125]', '[176,120,285,269]', '[1,204,178,413]', '[25,17,483,289]', '[10,10,200,150]', '[276,78,504,319]', '[94,0,504,364]', '[10,10][200,150]']

# new_pred = []

# for p in pred:
#     try:
#         new_pred.append(ast.literal_eval(p))
#     except:
#         new_pred.append(p)

# print(pred)
# print(new_pred)
