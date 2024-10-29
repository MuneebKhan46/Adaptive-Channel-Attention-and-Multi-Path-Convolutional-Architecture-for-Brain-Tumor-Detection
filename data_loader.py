import os
from PIL import Image
import numpy as np

def load_data(dataset_dir, classes, width, height):
    images, labels = [], []
    for i, class_name in enumerate(classes):
        class_path = os.path.join(dataset_dir, class_name)
        for img_name in os.listdir(class_path):
            try:
                img = Image.open(os.path.join(class_path, img_name)).convert('RGB').resize((width, height))
                images.append(np.array(img))
                labels.append(i)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
    return np.array(images), np.array(labels)
