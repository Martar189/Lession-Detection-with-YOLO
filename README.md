# EARLY INFECTION DETECTION USING YOLO

## MOTIVATION

Pin site infections are a common complication of external fixation that places a significant burden on the patient and healthcare system. Such infections increase the number of clinic visits required during a patient’s course of treatment, can result in the need for additional treatment including antibiotics and surgery, and most importantly can compromise patient outcomes should osteomyelitis or instability result from pin loosening or need for pin or complete construct removal. 

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4960058/


## GOAL
The goal of this project was to asses the performance of a custom version of YOLO (You only look once) model to support early infection detection.

***WHY YOLO?***

We were working under the hypothesis that the segmentation of an image containing an skin lession, isolating the image from the background, improves the accuracy of the classification. 

Therefore we needed to implement a model that would separated bounding boxes and associated class probabilities. And that is in a nut shell what YOLO does.

## TASKS

***TASK 1 – LESION BOUNDING***
Train a model to output the coordinates of the top-left and bottom-right vertices of a rectangle that contains the lesion area (bounding box)

***TASK 2 – LESION CLASSIFICATION***
Use a classifier that discriminates between benign and malign lesions


## PROJECT LIFECYCLE 

### DATA PREPROCESSING 

Image Classification problems are highly complex and we know that…

Complexity(Problem) ∝ Size(Data) ∝ Size(Model)

This is a challenge in medical applications, as we normally have very limited number of images

How do you overcome the problem? 

***DATA AUGMENTATION***





Initial set: 206 samples
Final dataset: Train on 1148 samples, validate on 42 samples

LEARNINGS
We had to rebuild the pre - processing function to apply data augmentation to the training set only to avoid leakage
We needed to shuffle the data to avoid any inference based on the order of the images

https://pjreddie.com/media/files/papers/yolo.pdf




