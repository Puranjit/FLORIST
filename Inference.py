# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 20:23:59 2025

@author: puran
"""

from joblib import load
import numpy as np
import pandas as pd

feature_embeddings = np.load('feature_embeddings.npy')

# Load the trained SVC model 
clf = load('svc_model.joblib')

# Use it to make predictions on the full dataset
y_pred = clf.predict(feature_embeddings)

# Load only the sheet named "Sheet2" from the Excel file
df = pd.read_excel('Flower classification project.xlsx', sheet_name='Full dataset')

# Creating a new column for saving the predictions into a new excel file
df['Predicted_Label'] = y_pred

# Save the updated DataFrame to a new Excel file
df.to_excel('Flower_classification_with_predictions.xlsx', index=False)