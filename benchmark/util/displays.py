from .bounding_boxes import *
from .benchmark_tools import *

def show_differences(inputs, predictions, input_normal=False):
    img_list, _, ref_list = split_inputs(inputs)
    for i in range(len(img_list)):
        new_img = draw_bboxes(img_list[i], ref_list[i], colour='red', normalised=input_normal)

        for key, val in predictions.items():
            if is_valid_bbox(val[i]):
                new_img = draw_bboxes(new_img, [val[i]], colour='blue', label=key, normalised=True)
            # new_img.show()

        display(new_img)


# for Colab
def show_individual(inputs, predictions, judge_evaluations=None):
    img_list, qn_list, ref_list = split_inputs(inputs)
    if len(qn_list) == 0: # only global questions, no individual
        qn_list = ["" for _ in img_list]

    for i in range(len(img_list)):
        display(img_list[i])
        print("Question: {}".format(qn_list[i]))
        print("Reference: {}".format(ref_list[i]))
        for key, val in predictions.items():
            print("Predicted ({}): {}".format(key, val[i]))
            if judge_evaluations is not None:
                judge_scores = judge_evaluations[key][0]
                judge_reasons = judge_evaluations[key][1]
                print("Rating ({}): {} ({})".format(key, judge_scores[i], judge_reasons[i]))

def show_results(inputs, predictions, evaluations):
    _, _, ref_list = split_inputs(inputs)
    print("Benchmark Results:")
    print("Truth: {}".format(ref_list))
    for key, val in evaluations.items():
        print("{}: {} ({})".format(key, val, predictions[key]))
