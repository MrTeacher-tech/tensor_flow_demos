import tensorflow as tf
from keras import layers, models

import create_datasets

# Check if TensorFlow is using the GPU
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")

# Define a function to create the model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),  # Define the input shape explicitly
        layers.Flatten(),  # No need for input_shape here
        layers.Dense(128, activation='relu'),  # Hidden layer
        layers.Dense(num_classes, activation='softmax')  # Output layer for classes
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def write_to_tfrecord(dataset, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for image, label in dataset:
            # Ensure the image is in the correct format
            image = tf.image.convert_image_dtype(image, tf.uint8)  # Convert to uint8
            image_encoded = tf.io.encode_jpeg(image)  # Encode image to JPEG

            image_encoded = tf.io.encode_jpeg(image)
            
            
            # Convert the Tensor to a TensorFlow Feature
            features = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            
            # Serialize and write the example to the TFRecord file
            writer.write(example.SerializeToString())

# Example use case (replace input_shape and num_classes with your own):
# For example, if your resized images are 128x128 and you have 2 classes (Indian food, barbecue food)
input_shape = (128, 128, 3)  # for color images, change to (128, 128, 1) if grayscale
num_classes = 2

IMAGE_SIZE = (128, 128)  # Resize all images to 128x128

IMAGE_DIR_0 = '../images/american_food/'  # Path to your directory with images
IMAGE_DIR_1 = '../images/indian_food/'  # Path to your directory with images

CLASS_LABEL_0 = 0  # Label for American food
CLASS_LABEL_1 = 1  # Label for Indian food

american_images, american_labels = create_datasets.create_dataset(IMAGE_DIR_0, CLASS_LABEL_0, IMAGE_SIZE)

dataset_american_food = tf.data.Dataset.from_tensor_slices((american_images, american_labels))

print(f"American Food dataset created with {len(american_images)} images.")

indian_images, indian_labels = create_datasets.create_dataset(IMAGE_DIR_1, CLASS_LABEL_1, IMAGE_SIZE)

dataset_indian_food = tf.data.Dataset.from_tensor_slices((indian_images, indian_labels))

print(f"Indian Food dataset created with {len(indian_images)} images.")

# Save datasets to TFRecord files
write_to_tfrecord(dataset_american_food, '../datasets/american_food_dataset.tfrecord')
write_to_tfrecord(dataset_indian_food, '../datasets/indian_food_dataset.tfrecord')

print("Datasets saved as TFRecord files.")


