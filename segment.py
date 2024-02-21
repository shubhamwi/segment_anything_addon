import os
import cv2
import numpy as np
import torch
import json
import argparse
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
from pycocotools import mask as maskUtils
from tqdm import tqdm 

class ObjectSegmentation:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_type = args.model_type
        self.checkpoint_path = args.checkpoint_path

        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)
        self.mask_predictor = SamPredictor(self.sam)

        self.images_folder = args.images_folder
        self.annotations_folder = args.annotations_folder
        self.output_folder = args.output_folder
        self.classes_file = args.classes_file

        with open(self.classes_file, 'r') as f:
            self.class_names = [line.strip() for line in f]

        os.makedirs(self.output_folder, exist_ok=True)
        self.annotations = []
        self.image_info = {}
        self.image_id = 1
        self.annotation_id = 1

    def load_and_process_image(self, image_file):
        try:
            image_path = os.path.join(self.images_folder, image_file)
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise Exception(f"Failed to load image: {image_file}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            return image_rgb
        except Exception as e:
            print(f"Error loading image: {image_file}")
            print(e)
            return None

    def process_annotations(self, image_file, image_rgb):
        mask_predictor = self.mask_predictor
        annotation_file = image_file.replace('.png', '.txt')
        annotation_path = os.path.join(self.annotations_folder, annotation_file)

        with open(annotation_path, 'r') as f:
            annotations_data = [line.strip().split() for line in f]

        overlay = image_rgb.copy()

        # Set the image for mask prediction
        mask_predictor.set_image(image_rgb)  # Set the image here

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

            score_threshold = 0.3
            binary_mask = masks[0] >= score_threshold
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx_contour) % 2 != 0:
                    approx_contour = np.concatenate((approx_contour, approx_contour[0:1]), axis=0)
                cv2.drawContours(overlay, [approx_contour], -1, (0, 255, 0), 2)
                class_name = self.class_names[int(label)] if int(label) < len(self.class_names) else "Unknown"
                text_position = (approx_contour[0][0][0], approx_contour[0][0][1] - 10)
                cv2.putText(overlay, class_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                segmentation = approx_contour.reshape(-1).tolist()

                if len(segmentation) % 2 != 0:
                    segmentation.append(segmentation[0])

                if len(segmentation) >= 6:
                    x = segmentation[::2]
                    y = segmentation[1::2]
                    if len(x) != len(y):
                        min_len = min(len(x), len(y))
                        x = x[:min_len]
                        y = y[:min_len]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    self.annotations.append({
                        'id': self.annotation_id,
                        'image_id': self.image_id,
                        'category_id': int(label) + 1,
                        'bbox': [x_min, y_min, box_width, box_height],
                        'area': int(area),
                        'iscrowd': 0,
                        'segmentation': [segmentation]
                    })
                    self.annotation_id += 1

        output_path = os.path.join(self.output_folder, image_file)
        # cv2.imwrite(output_path, overlay)
        self.image_info[self.image_id] = {
            'id': self.image_id,
            'file_name': image_file,
            'width': image_rgb.shape[1],
            'height': image_rgb.shape[0]
        }
        self.image_id += 1

    def create_coco_json(self):
        coco_data = {
            'images': list(self.image_info.values()),
            'type': 'instances',
            'annotations': self.annotations,
            'categories': [{'id': i + 1, 'name': class_name} for i, class_name in enumerate(self.class_names)]
        }
        coco_json_path = os.path.join(self.output_folder, 'coco_annotations.json')
        with open(coco_json_path, 'w') as f:
            json.dump(coco_data, f)
        print(f"Segmentation and annotation completed. Results saved to {self.output_folder}")

def main():
    parser = argparse.ArgumentParser(description='Object Segmentation with SAM')
    parser.add_argument('--model_type', type=str, default="vit_b", help='SAM model type')
    parser.add_argument('--checkpoint_path', type=str, default="/home/shubham/work/detection_models/cloud/sam/models/sam_vit_b_01ec64.pth", help='Path to SAM model checkpoint')
    parser.add_argument('--images_folder', type=str, required=True, help='Path to the images folder')
    parser.add_argument('--annotations_folder', type=str, required=True, help='Path to the annotations folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--classes_file', type=str, required=True, help='Path to the classes.txt file')

    args = parser.parse_args()
    segmentation = ObjectSegmentation(args)

    image_files = [file for file in os.listdir(args.images_folder) if not file.startswith('.ipynb_checkpoints')]
    for image_file in tqdm(image_files):
        image_rgb = segmentation.load_and_process_image(image_file)
        if image_rgb is not None:
            segmentation.process_annotations(image_file, image_rgb)
    
    segmentation.create_coco_json()

if __name__ == "__main__":
    main()

# python3 segment.py --images_folder /home/shubham/work/datasets/dalmiya_datasets/dalmia_2023_08_26_dataset/images --annotations_folder /home/shubham/work/datasets/dalmiya_datasets/dalmia_2023_08_26_dataset/labels --output_folder /home/shubham/work/datasets/dalmiya_datasets/dalmia_2023_08_26_dataset/annotation --classes_file /home/shubham/work/datasets/dalmiya_datasets/dalmia_2023_08_26_dataset/new_classes.txt