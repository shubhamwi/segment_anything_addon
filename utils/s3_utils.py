import boto3
import cv2
import datetime

def upload_image_to_s3(bucket_name, folder, filename, image_bytes):
    """
    Uploads an image to an S3 bucket.

    Args:
        bucket_name: The name of the S3 bucket.
        folder: The folder within the bucket to upload the image.
        filename: The name of the image file.
        image_bytes: The image data as bytes.

    """
    s3_client = boto3.client('s3')
    s3_client.put_object(
        Bucket=bucket_name,
        Key=f"{folder}/{filename}",
        Body=image_bytes
    )
    print("Image uploaded to S3:", filename)



def generate_filename():
    """
    Generates a unique filename based on the current date and time.

    Returns:
        The generated filename as a string.

    """
    current_datetime = datetime.datetime.now()
    current_datetime = current_datetime + datetime.timedelta(hours=5, minutes=30)
    filename = f"image_{current_datetime}.png"
    return filename
