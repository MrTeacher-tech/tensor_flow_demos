import os
import tensorflow as tf
from PIL import Image
import numpy as np


# Function to load and preprocess a single image
def load_and_preprocess_image(image_path, img_size):
    # Open the image
    img = Image.open(image_path)
     # Convert images with transparency to RGBA
    if img.mode == 'P' or (img.mode == 'L' and 'transparency' in img.info):
        img = img.convert('RGBA')
        
    # Resize the image
    img_resized = img.resize(img_size)
    # Convert to RGB if necessary
    img_resized = img_resized.convert('RGB')
    # Convert image to numpy array
    img_array = np.array(img_resized)
    # Normalize the pixel values (0-255 to 0-1)
    img_array = img_array / 255.0
    return img_array

# Function to flip image and return two versions (original and flipped)
def augment_image(img_array):
    # Convert the image array to a Tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    # Flip the image horizontally
    flipped_img_tensor = tf.image.flip_left_right(img_tensor)
    # Convert the tensor back to a numpy array
    return img_array, flipped_img_tensor.numpy()

# Process all images in the directory
def create_dataset(image_dir, label, img_size):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".webp"):
            image_path = os.path.join(image_dir, filename)
            img_array = load_and_preprocess_image(image_path, img_size)
            original_img, flipped_img = augment_image(img_array)
            images.extend([original_img, flipped_img])
            labels.extend([label, label])
    # Convert lists to numpy arrays
    images_np = np.array(images)
    labels_np = np.array(labels)
    return images_np, labels_np

# Create dataset for American food
#american_images, american_labels = create_dataset(IMAGE_DIR, CLASS_LABEL, IMG_SIZE)

# Example: Combine this dataset with another (e.g., Indian food)
# indian_images, indian_labels = create_dataset('./indian_food/', 1)  # Label 1 for Indian food
# full_dataset_images = np.concatenate((american_images, indian_images), axis=0)
# full_dataset_labels = np.concatenate((american_labels, indian_labels), axis=0)

# You can then convert this into a TensorFlow dataset if needed
#dataset = tf.data.Dataset.from_tensor_slices((american_images, american_labels))

#print(f"Dataset created with {len(american_images)} images.")
