import json

def load_coco_data(coco_file):
    """
    Loads COCO data from a JSON file.

    :param coco_file: Path to the COCO JSON file.
    :return: Loaded COCO data as a dictionary.
    """
    with open(coco_file, 'r') as file:
        data = json.load(file)
    return data

def extract_data(data):
    """
    Extracts images, annotations, and categories from COCO data.

    :param data: COCO data as a dictionary.
    :return: Tuple (images, annotations, categories) containing the extracted data.
    """
    images = data['images']
    annotations = data['annotations']
    categories = {category['id']: category['name'] for category in data['categories']}
    return images, annotations, categories
