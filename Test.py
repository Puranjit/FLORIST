# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 21:06:36 2025

@author: puran
"""
# Importing libraries
import numpy as np
import torch
from PIL import Image
import os
from torchvision import transforms
from tqdm.notebook import tqdm
from joblib import load
from PIL import ImageFile

# TESTING
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

cwd = os.getcwd()

# Load and preprocess the test dataset
test_labels = {}  # Dictionary to hold test file paths and their labels
# TEST_DIR = os.path.join(cwd, "Images/test")
TEST_DIR = os.path.join(cwd, "Testing images")

# Reading images from Testing dataset
for folder in os.listdir(TEST_DIR):
    for file in os.listdir(os.path.join(TEST_DIR, folder)):
        if file.endswith(".jpg"):
            full_name = os.path.join(TEST_DIR, folder, file)
            test_labels[full_name] = folder

test_files = list(test_labels.keys())

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Transforming images into format acceptable by Dinov2 model
transform_image = transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Resize((840, 840)), 
                                    transforms.Normalize([0.5], [0.5])
                                    ])

def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    try:
        img = Image.open(img)
        transformed_img = transform_image(img)[:3].unsqueeze(0)
        return transformed_img
    except Exception as e:
        print(f"Error loading image {img}: {e}")
        return None

# Extract embeddings for the test dataset
test_embeddings = []
test_y = []

# Extracting feature embeddings using Dinov2 - vision foundation model

# dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14") # 21 M
# dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') # 86 M
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') # 300 M
# dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') # 300 M

# Running code on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2.to(device)

# Extracting feature embeddings of images from the Testing dataset
# We will use the trained SVC model to perform flower classifications using teh feature embeddings from testing dataset
# Extract embeddings with a progress bar
with torch.no_grad():
    for file in tqdm(test_files, desc="Processing Images", unit="image"):
        img_tensor = load_image(file)
        if img_tensor is None:  # Skip corrupted images
            continue
        embedding = dinov2(img_tensor.to(device))
        test_embeddings.append(np.array(embedding[0].cpu()).reshape(1024))  # Flatten to 1D
        test_y.append(test_labels[file])  # Ground truth label

# Ground truth Testing labels        
testing_labels = list(test_labels.values())
testing_labels = np.array(testing_labels)

# Convert embeddings to NumPy array
test_embeddings = np.array(test_embeddings)

# Loading the trained flower classification ML model
clf = load('svc_model.joblib')
    
# Predict using the trained model
test_predictions = clf.predict(test_embeddings)

# Generate confusion matrix
cm = confusion_matrix(test_y, test_predictions, labels=list(set(test_y)), normalize='true')
# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(test_y)))
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Generating evaluation metric scores on Testing dataset
          
print(classification_report(test_y, test_predictions, labels=list(set(test_y))))

print(f1_score(test_y, test_predictions, average="weighted")) # "macro"
print(precision_score(test_y, test_predictions, average="weighted")) # "macro"
print(recall_score(test_y, test_predictions, average="weighted")) # "macro"


# We can also train multiple Regression-based ML models using 'LazyPredict' library
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(np.array(embedding_list).reshape(-1, 1024), test_embeddings, y, test_y)
models
