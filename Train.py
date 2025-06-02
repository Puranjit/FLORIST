# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 21:06:00 2025

@author: puran
"""

# Importing libraries
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import json
from tqdm.notebook import tqdm
from joblib import dump
# from dinov2.models import vision_transformer as vits

cwd = os.getcwd()

# Defining path to the dataset including training images
ROOT_DIR = os.path.join(cwd, "Training images")

labels = {}

# Reading images
for folder in os.listdir(ROOT_DIR):
    print(folder)
    for file in os.listdir(os.path.join(ROOT_DIR, folder)):
        if file.endswith(".jpg"):
            full_name = os.path.join(ROOT_DIR, folder, file)
            labels[full_name] = folder

files = labels.keys()

# Extracting feature embeddings using Dinov2 - vision foundation model

# dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14") # 21 M
# dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') # 86 M
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') # 300 M
# dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') # 300 M

# Running code on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2.to(device)

# Transforming images into format acceptable by Dinov2 model
transform_image = transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Resize((840, 840)), 
                                    transforms.Normalize([0.5], [0.5])
                                    ])
                            
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to load images from the training dataset
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

# Function to compute feature embeddings from images in the training dataset    
def compute_embeddings(files: list) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = {}
    
    with torch.no_grad():
      for i, file in enumerate(tqdm(files)):
        embeddings = dinov2(load_image(file).to(device))

        all_embeddings[file] = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()

    with open("all_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    return all_embeddings

# Compute embeddings and store them in a DataFrame
embeddings = compute_embeddings(files)

# Ground truth labels
y = [labels[file] for file in files]

embedding_list = list(embeddings.values())

# Training Machine Learning model
from sklearn import svm

# Define the Support Vector Classifier ML model
clf = svm.SVC(gamma='scale', class_weight='balanced')

# Training the flower classification ML model
clf.fit(np.array(embedding_list).reshape(-1, 1024), y)

# Saving the trained ML model
dump(clf, 'svc_model.joblib')
