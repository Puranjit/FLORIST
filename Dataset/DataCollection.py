# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:48:33 2024

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

flower_folder = "Training images/Flower"
non_flower_folder = "Training images/Non Flower"

os.makedirs(flower_folder, exist_ok=True)
os.makedirs(non_flower_folder, exist_ok=True)

# Initialize variables
current_species = None  # To track the current species
count = 1  # Start count from 1

# Download each image and save it with the specified naming format
for index, row in df.iterrows():
    species = row["Species"]
    gbifID = row["gbifID"]
    url = row["url_ori"]
    fl_status = row["FL_status"]
    
    # Check if species has changed, reset count if it has
    if species != current_species:
        current_species = species
        count = 1  # Reset the counter for the new species
    
    # Determine FL_status suffix (1 for 'FL', 0 otherwise)
    fl_suffix = "_1" if fl_status == "Flower" else "_0"
    # fl_suffix = "_1" if fl_status == "FL" else "_0"
    # Determine the target folder based on FL_status
    
    target_folder = flower_folder if fl_suffix == "_1" else non_flower_folder
    # Skip if the URL is missing or invalid
    
    if pd.isna(url) or not isinstance(url, str):
        continue
    
    # Generate the image filename
    filename = f"{species}_{gbifID}_{count}{fl_suffix}.jpg"
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
