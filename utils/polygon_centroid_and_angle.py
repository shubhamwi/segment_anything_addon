import os
import cv2
import numpy as np
import math
import argparse
from image_utils import read_image, save_image, draw_polygon
from math_utils import perform_eigen_decomposition, calculate_centroid

class ImageProcessor:
    def __init__(self, image_path, annotation_path):
        """
        Initialize the ImageProcessor object.

        Args:
            image_path (str): Path to the image file.
            annotation_path (str): Path to the annotation file.
        """
        self.image_path = image_path
        self.annotation_path = annotation_path
    
    def normalize_polygons(self, polygons, img_w, img_h):
        """
        Normalize the coordinates of polygons.

        Args:
            polygons (list): List of polygons represented as numpy arrays of shape (n, 2).
            img_w (int): Width of the image.
            img_h (int): Height of the image.
        """
        for polygon in polygons:
            polygon[:, 0] = (polygon[:, 0] * img_w).astype(int)
            polygon[:, 1] = (polygon[:, 1] * img_h).astype(int)

    def read_image_label(self):
        """
        Read the image and annotation file.

        Returns:
            tuple: A tuple containing the image as a numpy array and the list of polygons.
        """
        # Read image
        #image = cv2.imread(self.image_path)
        image = read_image(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        # Read .txt file for this image
        with open(self.annotation_path, "r") as f:
            txt_lines = f.readlines()
            polygons = []
            for line in txt_lines:
                cls_idx, *coords = line.split()
                polygon = np.array([[float(coords[i]), float(coords[i + 1])] for i in range(0, len(coords), 2)])
                polygons.append(polygon)

        # Scale polygon coordinates to the image dimensions
        self.normalize_polygons(polygons, img_w, img_h)

        return image, polygons

    def draw_polygon_and_centroid(self, image, polygon, centroid):
        """
        Draw the polygon and centroid on the image.

        Args:
            image (numpy.ndarray): The image as a numpy array.
            polygon (numpy.ndarray): The polygon coordinates as a numpy array of shape (n, 2).
            centroid (tuple): The centroid coordinates as a tuple (x, y).
        """
        # Draw polygon on the image
        #cv2.polylines(image, [polygon], isClosed=False, color=(0, 255, 0), thickness=2)
        draw_polygon(image, polygon, (0, 255, 0))
        # Draw centroid on the image
        cv2.circle(image, centroid, 5, (255, 255, 255), -1)
        cv2.putText(image, "centroid", (centroid[0] - 25, centroid[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

    def calculate_eigenvectors(self, centered_x, centered_y):
        """
        Calculate the eigenvectors of the covariance matrix.

        Args:
            centered_x (numpy.ndarray): The x-coordinate values centered around the centroid.
            centered_y (numpy.ndarray): The y-coordinate values centered around the centroid.

        Returns:
            tuple: A tuple containing the eigenvalues and eigenvectors.
        """
        eigenvalues, eigenvectors = perform_eigen_decomposition(centered_x, centered_y)

        # Sort eigenvalues and eigenvectors in descending order
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        return eigenvalues, eigenvectors

    def draw_angle_line(self, image, cX, cY, angle_rad, angle_wrt_horizontal):
        """
        Draw the angle line and the horizontal line passing through the centroid on the image.

        Args:
            image (numpy.ndarray): The image as a numpy array.
            cX (int): The x-coordinate of the centroid.
            cY (int): The y-coordinate of the centroid.
            angle_rad (float): The angle in radians.
            angle_wrt_horizontal (float): The angle with respect to the horizontal line passing through the centroid.
        """
        # Calculate start and end points for angle line
        line_length = min(image.shape[0], image.shape[1]) // 2
        angle_line_start = (
            int(cX + line_length * math.cos(angle_rad)),
            int(cY + line_length * math.sin(angle_rad))
        )
        angle_line_end = (
            int(cX - line_length * math.cos(angle_rad)),
            int(cY - line_length * math.sin(angle_rad))
        )

        # Calculate start and end points for horizontal line passing through centroid
        horizontal_line_start = (int(cX - line_length), cY)
        horizontal_line_end = (int(cX + line_length), cY)

        # Draw angle line on the image
        cv2.line(image, angle_line_start, angle_line_end, (0, 0, 255), 2)

        # Draw horizontal line on the image
        cv2.line(image, horizontal_line_start, horizontal_line_end, (255, 0, 0), 2)

        # Display the angle value
        angle_text = f"Angle: {angle_wrt_horizontal:.2f} degrees"
        text_size, _ = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_width, text_height = text_size
        text_x = cX - text_width // 2
        text_y = cY + text_height // 2
        cv2.putText(image, angle_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def process_image(self, image, polygons):
        """
        Process the image by drawing polygons, centroids, and angle lines.

        Args:
            image (numpy.ndarray): The image as a numpy array.
            polygons (list): List of polygons represented as numpy arrays of shape (n, 2).
        """
        for coords in polygons:
            # Convert polygon coordinates to pixel coordinates
            pixel_coords = np.array([[coord[0], coord[1]] for coord in coords], dtype=np.int32)

            # Calculate the centroid of the polygon
            cX, cY = calculate_centroid(pixel_coords)

            self.draw_polygon_and_centroid(image, pixel_coords, (cX, cY))

            # Separate X and Y coordinates
            x_coords = pixel_coords[:, 0]
            y_coords = pixel_coords[:, 1]

            # Calculate the centroid-subtracted coordinates
            centered_x = x_coords - cX
            centered_y = y_coords - cY

            eigenvalues, eigenvectors = self.calculate_eigenvectors(centered_x, centered_y)

            # Get the eigenvector corresponding to the largest eigenvalue
            max_eigenvector = eigenvectors[:, 0]

            # Calculate the angle between the eigenvector and the horizontal line passing through the centroid
            angle_rad = math.atan2(max_eigenvector[1], max_eigenvector[0])

            # Convert angle to degrees
            angle_deg = math.degrees(angle_rad)

            # Calculate the angle between the angle line and the horizontal 180-degree line passing through the centroid
            angle_wrt_horizontal = 180 - angle_deg if angle_deg > 0 else -angle_deg

            self.draw_angle_line(image, cX, cY, angle_rad, angle_wrt_horizontal)

    def save_image(self, image, output_path):
        """
        Save the image to the specified output path.

        Args:
            image (numpy.ndarray): The image as a numpy array.
            output_path (str): Path to save the image.
        """
        #cv2.imwrite(output_path, image)
        save_image(image, output_path)


def main():
    parser = argparse.ArgumentParser(description="Image Processing")
    parser.add_argument("--images_folder", required=True, help="path to the folder containing input images")
    parser.add_argument("--annotations_folder", required=True, help="path to the folder containing annotations")
    parser.add_argument("--output_folder", required=True, help="path to the output folder")
    args = parser.parse_args()

    images_folder = args.images_folder
    annotations_folder = args.annotations_folder
    output_folder = args.output_folder

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each image and annotation file in the folders
    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            annotation_path = os.path.join(annotations_folder, os.path.splitext(filename)[0] + ".txt")

            # Create an ImageProcessor instance
            processor = ImageProcessor(image_path, annotation_path)

            # Read image and annotation file
            image, polygons = processor.read_image_label()

            # Process the image
            processor.process_image(image, polygons)

            # Save the processed image
            output_path = os.path.join(output_folder, filename)
            processor.save_image(image, output_path)

            print(f"Processed {filename} and saved to {output_path}")


if __name__ == "__main__":
    main()