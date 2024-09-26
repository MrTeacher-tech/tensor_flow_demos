import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models
import numpy as np
import os

import process_images as P_I

CLASS_LABEL_0 = 0  # Label for American food
CLASS_LABEL_1 = 1  # Label for Indian food

# Check if TensorFlow is using the GPU
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")

# Function to parse TFRecord dataset
def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),  # Image feature
        'label': tf.io.FixedLenFeature([], tf.int64)    # Label feature
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    # Decode and preprocess image
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    image = tf.image.resize(image, [128, 128])  # Resize to (128, 128)
    
    label = example['label']
    return image, label

# Load American food dataset
dataset_american_food = tf.data.TFRecordDataset('../datasets/american_food_dataset.tfrecord')
dataset_american_food = dataset_american_food.map(parse_tfrecord_fn)

# Load Indian food dataset
dataset_indian_food = tf.data.TFRecordDataset('../datasets/indian_food_dataset.tfrecord')
dataset_indian_food = dataset_indian_food.map(parse_tfrecord_fn)

# Combine the two datasets
combined_dataset = dataset_american_food.concatenate(dataset_indian_food)

# Shuffle and batch the data
BATCH_SIZE = 32
combined_dataset = combined_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)

# Define the model architecture
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),  # Define the input shape explicitly
        layers.Flatten(),                 # Flatten the input image
        layers.Dense(128, activation='relu'),  # Hidden layer
        layers.Dense(num_classes, activation='softmax')  # Output layer for 2 classes
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Define the input shape and number of classes
input_shape = (128, 128, 3)  # For color images
num_classes = 2  # American food and Indian food

# Create the model
model = create_model(input_shape, num_classes)

# Train the model
EPOCHS = 17
model.fit(combined_dataset, epochs=EPOCHS)

model.save("../models/model_128x128_17e.keras")

# Evaluate the model (you can load a separate test dataset for evaluation)
print("Evaluation on training data:")
test_loss, test_acc = model.evaluate(combined_dataset)
print(f'\nTest accuracy: {test_acc}')








