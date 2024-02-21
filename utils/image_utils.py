import cv2

def read_image(filename):
    """
    Reads an image from a file.

    :param filename: Path to the image file.
    :return: Loaded image as a NumPy array.
    """
    image = cv2.imread(filename)
    return image


def encode_image(img):
    """
    Encodes an image as PNG format.

    Args:
        img: The image data as a numpy array.

    Returns:
        The image data as bytes in PNG format.

    """
    _, image_png = cv2.imencode('.png', img)
    image_bytes = image_png.tobytes()
    return image_bytes


def save_image(image, filename):
    """
    Saves an image to a file.

    :param image: Image to be saved, as a NumPy array.
    :param filename: Path to save the image file.
    """
    cv2.imwrite(filename, image)
    
def show_image(img):
    """
    Displays the image in a window named 'Camera Feed'.
    
    Args:
        img: The image data as a numpy array.
    """
    cv2.imshow('Camera Feed', img)
    cv2.waitKey(1)

def draw_polygon(image, vertices, color):
    """
    Draws a polygon on the image.

    :param image: Image on which the polygon will be drawn, as a NumPy array.
    :param vertices: List of vertices of the polygon.
    :param color: Color of the polygon (BGR format).
    """
    cv2.polylines(image, [vertices], True, color, thickness=2)