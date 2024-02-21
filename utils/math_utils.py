import numpy as np
import cv2

def xywh_to_xyxy(x_center, y_center, box_width, box_height, image_width, image_height):
    """
    Converts bounding box coordinates from (x_center, y_center, width, height) format to (x_min, y_min, x_max, y_max) format.

    :param x_center: The x-coordinate of the center of the bounding box.
    :param y_center: The y-coordinate of the center of the bounding box.
    :param box_width: The width of the bounding box.
    :param box_height: The height of the bounding box.
    :param image_width: The width of the image containing the bounding box.
    :param image_height: The height of the image containing the bounding box.
    :return: Tuple (x_min, y_min, x_max, y_max) representing the converted bounding box coordinates.
    """
    x_min = int((x_center - box_width / 2) * image_width)
    y_min = int((y_center - box_height / 2) * image_height)
    x_max = int((x_center + box_width / 2) * image_width)
    y_max = int((y_center + box_height / 2) * image_height)
    return x_min, y_min, x_max, y_max

def perform_eigen_decomposition(centered_x, centered_y):
    """
    Perform eigendecomposition on the covariance matrix.

    Args:
        centered_x (numpy.ndarray): The x-coordinate values centered around the centroid.
        centered_y (numpy.ndarray): The y-coordinate values centered around the centroid.

    Returns:
        tuple: A tuple containing the eigenvalues and eigenvectors.
    """
    # Construct the covariance matrix
    covariance_matrix = np.cov(centered_x, centered_y)

    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    return eigenvalues, eigenvectors

def calculate_centroid(polygon):
    """
    Calculate the centroid of a polygon.

    Args:
        polygon (numpy.ndarray): The polygon coordinates as a numpy array of shape (n, 2).

    Returns:
        tuple: A tuple containing the x and y coordinates of the centroid.
    """
    moments = cv2.moments(polygon)
    cX = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else 0
    cY = int(moments["m01"] / moments["m00"]) if moments["m00"] != 0 else 0
    return cX, cY