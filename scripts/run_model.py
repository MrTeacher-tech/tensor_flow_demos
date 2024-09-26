import process_images as P_I
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models
import numpy as np

CLASS_LABEL_0 = 0  # Label for American food
CLASS_LABEL_1 = 1  # Label for Indian food

model = models.load_model('../models/model_128x128_17e.keras')

# Example prediction on a new image (replace with your test image paths)
test_images_dir = "../images/test_images/" 



# Loop through all images in the test images folder
for filename in os.listdir(test_images_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        image_path = os.path.join(test_images_dir, filename)
        processed_img = P_I.preprocess_image(image_path)
        
        # Here you can make predictions using your model
        predictions = model.predict(processed_img)
        predicted_class = tf.argmax(predictions, axis=-1).numpy()

        if predicted_class[0] == CLASS_LABEL_0:
            predicted_class_str = 'American Food'

        elif predicted_class[0] == CLASS_LABEL_1:
            predicted_class_str = 'Indian Food'

        
        
        print(f"Image: {filename}, Predicted Class: {predicted_class_str}")

        # Display the image and prediction result
        plt.imshow(np.squeeze(processed_img))  # Remove batch dimension for display
        plt.title(f"Predicted Class: {predicted_class_str}")
        plt.axis('off')
        plt.show()

        