import os
from PIL import Image
import numpy as np
import tensorflow as tf

# Function to preprocess a single image
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize to [0, 1]
    img = tf.image.resize(img, [128, 128])  # Resize to match model input
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

def calculate_average_image_size(image_dir):
    widths = []
    heights = []
    
    # Loop through all the images in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpeg", ".jpg", ".webp")):
            image_path = os.path.join(image_dir, filename)
            with Image.open(image_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
    
    # Calculate average width and height
    if widths and heights:
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        return avg_width, avg_height
    else:
        raise ValueError("No valid images found in the directory.")

training_dirs = ['../images/american_food', '../images/indian_food']

widths = []
heights = []
for dir in training_dirs:
    try:
        # Calculate average image size
        average_width, average_height = calculate_average_image_size(dir)
        '''
        print("Dir:", dir)
        print(f"Average Image Width: {average_width:.2f}")
        print(f"Average Image Height: {average_height:.2f}")
        '''

        widths.append(average_width)
        heights.append(average_height)

    except ValueError as e:
        print(f"Error processing {dir}: {e}")

avg_avg_width = sum(widths)/len(widths)
avg_avg_height = sum(heights)/len(heights)
print(f"Total Average Image Width: {avg_avg_width:.2f}")
print(f"Total Average Image Height: {avg_avg_height:.2f}")

