
from PIL import ImageDraw

# when coordinates must be ordered (smaller first)
def fix_bbox(bbox):
    x1, y1, x2, y2 = bbox

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    return [x1, y1, x2, y2]

def draw_bboxes(image, bbox_list, colour='red', label='', normalised=False):
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    img_width, img_height = new_image.size

    for bbox in bbox_list:
        text_x = bbox[0]
        text_y = bbox[1]

        if normalised:
            bbox = [bbox[0]*img_width, bbox[1]*img_height, bbox[2]*img_width, bbox[3]*img_height]
            text_x = bbox[0]*img_width
            text_y = bbox[1]*img_height
        draw.rectangle(bbox, outline=colour, width=3)
        draw.text((text_x, text_y), label, fill=colour)
    return new_image

def eval_bbox(ref_list, img_list, pred_list):
    eval = []
    for i in range(len(ref_list)):
        ref_bbox = ref_list[i]

        img = img_list[i]
        img_width, img_height = img.size

        pred_bbox = pred_list[i] # normalised values

        if not is_valid_bbox(pred_bbox):
            iou = 0
        else:
            pred_bbox = [pred_bbox[0]*img_width, pred_bbox[1]*img_height, pred_bbox[2]*img_width, pred_bbox[3]*img_height]

            iou = intersection_over_union(ref_bbox, pred_bbox)
            iou = round(iou, 2)
        eval.append(iou)
    
    return eval

# bbox should be list of format: [x1, y1, x2, y2]
def is_valid_bbox(bbox):
    if not isinstance(bbox, list): return False
    if len(bbox) != 4: return False
    for x in bbox:
        if not isinstance(x, float): return False
    return True

# taken from LLM
def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    # The width and height must be positive, so we use max(0, ...)
    # to handle cases where there is no overlap.
    intersection_width = max(0, xB - xA)
    intersection_height = max(0, yB - yA)
    intersection_area = intersection_width * intersection_height

    # Compute the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the area of the union: area(A) + area(B) - area(intersection)
    union_area = float(boxA_area + boxB_area - intersection_area)

    # Compute the intersection over union
    # Handle the case of division by zero if union_area is 0
    iou = intersection_area / union_area if union_area != 0 else 0

    return iou