import os
import cv2
import numpy as np
import torch
import json
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
from pycocotools import mask as maskUtils
# Initialize the SAM model and predictor
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = '/home/shubham/work/detection_models/cloud/sam/models/sam_vit_b_01ec64.pth'
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_predictor = SamPredictor(sam)
# Set the paths to the images and annotations folders
IMAGES_FOLDER = '/home/shubham/work/datasets/object_detection_dataset/dataset_dummy/images'
ANNOTATIONS_FOLDER = '/home/shubham/work/datasets/object_detection_dataset/dataset_dummy/labels'
OUTPUT_FOLDER = '/home/shubham/work/datasets/object_detection_dataset/dataset_dummy/sam_data '
CLASSES_FILE = '/home/shubham/work/datasets/object_detection_dataset/dataset_dummy/txt_files/classes.txt'  # Replace with the actual path to your classes.txt file
# Load the class names from the classes.txt file
with open(CLASSES_FILE, 'r') as f:
    class_names = [line.strip() for line in f]
# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# Create a list to store the COCO annotations
annotations = []
# Create a dictionary to store image information
image_info = {}
# Initialize counters for image and annotation IDs
image_id = 1
annotation_id = 1
# Iterate over the image files
image_files = [file for file in os.listdir(IMAGES_FOLDER) if not file.startswith('.ipynb_checkpoints')]
for image_file in image_files:
    try:
        # Load the image
        image_path = os.path.join(IMAGES_FOLDER, image_file)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise Exception(f"Failed to load image: {image_file}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {image_file}")
        print(e)
        continue
    # Set the image for mask prediction
    mask_predictor.set_image(image_rgb)
    # Load the corresponding annotation file
    annotation_file = image_file.replace('.png', '.txt')
    annotation_path = os.path.join(ANNOTATIONS_FOLDER, annotation_file)
    with open(annotation_path, 'r') as f:
        annotations_data = [line.strip().split() for line in f]
    # Create a copy of the image for overlaying polygons
    overlay = image_rgb.copy()
    for annotation in annotations_data:
        label, x_center, y_center, box_width, box_height = map(float, annotation)
        x_min = int((x_center - box_width / 2) * image_rgb.shape[1])
        y_min = int((y_center - box_height / 2) * image_rgb.shape[0])
        x_max = int((x_center + box_width / 2) * image_rgb.shape[1])
        y_max = int((y_center + box_height / 2) * image_rgb.shape[0])
        box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits = mask_predictor.predict(
            box=box,
            multimask_output=True
        )
        # Adjust the score threshold
        score_threshold = 0.3  # Adjust this threshold
        binary_mask = masks[0] >= score_threshold
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate over the contours and draw polygons on the image
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            # Make the number of contour points even
            if len(approx_contour) % 2 != 0:
                approx_contour = np.concatenate((approx_contour, approx_contour[0:1]), axis=0)
            cv2.drawContours(overlay, [approx_contour], -1, (0, 255, 0), 2)
            # Draw class name on the image
            class_name = class_names[int(label)] if int(label) < len(class_names) else "Unknown"
            text_position = (approx_contour[0][0][0], approx_contour[0][0][1] - 10)
            cv2.putText(overlay, class_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Extract polygon coordinates and store them
            segmentation = approx_contour.reshape(-1).tolist()
            # Ensure segmentation points are even by completing the polygon
            if len(segmentation) % 2 != 0:
                segmentation.append(segmentation[0])
            # Ensure a minimum of three (x, y) coordinate pairs for a valid polygon
            if len(segmentation) >= 6:
                # Calculate area using shoelace formula
                x = segmentation[::2]
                y = segmentation[1::2]
                if len(x) != len(y):
                    # Adjust the number of points to make them equal
                    min_len = min(len(x), len(y))
                    x = x[:min_len]
                    y = y[:min_len]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                annotations.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': int(label) + 1,  # Increment the category ID by 1
                    'bbox': [x_min, y_min, box_width, box_height],
                    'area': int(area),
                    'iscrowd': 0,
                    'segmentation': [segmentation]
                })
                # Increment the annotation ID
                annotation_id += 1
    # Save the image with polygons overlay
    output_path = os.path.join(OUTPUT_FOLDER, image_file)
    cv2.imwrite(output_path, overlay)
    # Store image information
    image_info[image_id] = {
        'id': image_id,
        'file_name': image_file,
        'width': image_rgb.shape[1],
        'height': image_rgb.shape[0]
    }
    # Increment the image ID
    image_id += 1
# Create the COCO JSON structure
coco_data = {
    'images': list(image_info.values()),
    'type': 'instances',
    'annotations': annotations,
    'categories': [{'id': i + 1, 'name': class_name} for i, class_name in enumerate(class_names)]
}
# Save the COCO JSON file
coco_json_path = os.path.join(OUTPUT_FOLDER, 'coco_annotations.json')
with open(coco_json_path, 'w') as f:
    json.dump(coco_data, f)
print(f"Segmentation and annotation completed. Results saved to {OUTPUT_FOLDER}.")