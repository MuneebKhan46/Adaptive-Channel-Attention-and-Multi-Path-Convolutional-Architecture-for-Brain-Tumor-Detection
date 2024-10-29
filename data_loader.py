import os
from PIL import Image
import numpy as np

def load_data(dataset_dir, classes, width=224, height=224):
    """
    Load images and their corresponding labels from the dataset directory.

    Args:
        dataset_dir (str): Path to the dataset folder.
        classes (list): List of class names.
        width (int): Width to resize images.
        height (int): Height to resize images.

    Returns:
        Tuple: Numpy arrays containing images and their labels.
    """
    images, labels = [], []

    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(dataset_dir, class_name)
        for img_name in os.listdir(class_path):
            try:
                img = Image.open(os.path.join(class_path, img_name)).convert('RGB')
                img = img.resize((width, height))
                images.append(np.array(img))
                labels.append(class_index)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")

    return np.array(images), np.array(labels)
