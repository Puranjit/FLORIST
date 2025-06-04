# -*- coding: utf-8 -*-
"""
Created on Tue Jun 3 13:06:07 2025

@author: puran
"""

# FINAL CODE TO DOWNLOAD IMAGES
import pandas as pd
import os
import requests
from openpyxl import load_workbook


# Read the Excel file
file_path = "Flower classification project.xlsx"  # Replace with your Excel file path
df = pd.read_excel(file_path)

# Initialize variables
current_species = None  # To track the current species
count = 1  # Start count from 1

# Download each image and save it with the specified naming format
for index, row in df.iterrows():
    species = row["Species"]
    gbifID = row["gbifID"]
    url = row["url_ori"]
    
    # Check if species has changed, reset count if it has
    if species != current_species:
        current_species = species
        count = 1  # Reset the counter for the new species

    if pd.isna(url) or not isinstance(url, str):
        continue
    
    target_folder = "Flower classification dataset"
    
    # Generate the image filename
    filename = f"{species}_{gbifID}_{count}.jpg"
    filepath = os.path.join(target_folder, filename)

    try:
        # Download the image
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for errors

        # Save the image
        with open(filepath, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded: {filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

    count += 1  # Increment the count for the current species
    
    
# Generating feature embeddings of all images from our Flower classification dataset
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import json
from tqdm.notebook import tqdm
from dinov2.models import vision_transformer as vits

cwd = os.getcwd()

# Defining path to the dataset including training images
ROOT_DIR = os.path.join(cwd, "Flower classification dataset")

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

# Function to load images from the Flower classification  dataset
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

# Function to compute feature embeddings from images in the Flower classification dataset    
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
feature_embeddings = compute_embeddings(files)

# Saving feature embeddings of all the images in flower classification dataset
np.save('feature_embeddings.npy', feature_embeddings)