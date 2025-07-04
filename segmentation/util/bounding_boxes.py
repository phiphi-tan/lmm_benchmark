import torchvision
from torchvision.utils import draw_bounding_boxes

def display_results(image, results):
    
    print(results)
    results_dict = results[0]
    print(results_dict)

    colours = ['red', 'blue', 'green', 'yellow', 'white', 'grey'] # hardcoded colour list for now

    image_tensor = torchvision.transforms.ToTensor()(image)
    img = draw_bounding_boxes(image_tensor, results_dict['boxes'], results_dict['text_labels'])
    img = torchvision.transforms.ToPILImage()(img)
    img.show()

    return img