# FLORIST: Flower Label-maker from Open source Research grade Imagery via Self supervised Transformer 
This repository contains the source code and instruction for running: FLORIST: Flower Labelling from Open-source Research-grade Imagery via Self-supervised Transformer.


## Citizen science observations
Switchgrass, big bluestem, little bluestem, and Indiangrass are four major warm-season perennial grasses of the North American prairie. Research-grade photos of these four species with observation dates before January 1, 2023 were downloaded from the Global Biodiversity Information Facility repository (GBIF, www.GBIF.org) using the filters of “Scientific name”, “Present” for Occurrence status, “Human observation” for Basis of record, and “United States of America” for Country or area. A total of 43,861 photos (8,248 of switchgrass [doi: 10.15468/dl.xhrwtk], 11,081 of big bluestem [doi: 10.15468/dl.2j3v6c], 12,465 of Indiangrass [doi: 10.15468/dl.yvurbw], and 12,067 of little bluestem [doi: 10.15468/dl.unst7e]) were obtained. Because multiple images could be taken of the same plant from different angles and/or distances, we defined one location-date combination as one event when evaluating the latitudinal trend.

## FLORIST to classify flowering and non-flowering photos
We developed a computer vision AI named FLORIST to efficiently and effectively screen large numbers of photos taken by citizen scientists to identifying ones with fresh anthers/stigmas (Fig. S2). FLORIST consists of the pretrained DINOv215 and the machine learning classifier library. First, FLORIST used the pretrained DINOv2 ViT-L/14 distilled vision transformer (300 million parameters) to extract 1,024 feature embeddings from each photo. Next, FLORIST was trained to classify flowering or non-flowering images based on these 1,024 feature embeddings through a small dataset consisting of 320 images including an equal number of flowering and non-flowering photos from each species. FORIST evaluated 26 machine learning models included in the Lazy Predict library (https://lazypredict.readthedocs.io/en/latest/index.html) using a testing set consisting of randomly selected 1,000 photos (manually verified for visible anthers/stigmas) to identify the best performing model. Three standard metrics for classification , , and F1-score , where TP is true positive, FP is false positive, and FN is false negative, were calculated for both categories47. As FLORIST performed better in classifying the non-flowering category, after FLORIST filtered out non-flowering photos, we manually verified remaining photos for fresh anthers and/or stigmas.

# Method
![image](https://github.com/user-attachments/assets/dfe5c489-f717-44b3-81f9-cff16164db6c)


# Install
Pip install all requirements in a Python>=3.8 environment with PyTorch>=1.8

# Datasets
The Flower classification datasets can be downloaded from <Link>
