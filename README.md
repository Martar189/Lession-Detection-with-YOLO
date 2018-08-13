# EARLY INFECTION DETECTION USING YOLO

## MOTIVATION

Pin site infections are a common complication of external fixation that places a significant burden on the patient and healthcare system. Such infections increase the number of clinic visits required during a patient’s course of treatment, can result in the need for additional treatment including antibiotics and surgery, and most importantly can compromise patient outcomes should osteomyelitis or instability result from pin loosening or need for pin or complete construct removal. 

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4960058/


## GOAL
The goal of this project was to asses the performance of a custom version of YOLO (You only look once) model to support early infection detection.

***WHY YOLO?***

We were working under the hypothesis that the segmentation of an image containing an skin lession, isolating the image from the background, improves the accuracy of the classification. 

Therefore we needed to implement a model that would separated bounding boxes and associated class probabilities. And that is in a nut shell what YOLO does.

![YOLO](https://cdn-images-1.medium.com/max/1600/1*oOD2qLugrH-oc6EEKLzdBg.jpeg)


## TASKS

***TASK 1 – LESION BOUNDING***

Train a model to output the coordinates of the top-left and bottom-right vertices of a rectangle that contains the lesion area (bounding box)

***TASK 2 – LESION CLASSIFICATION***

Use a classifier that discriminates between benign and malign lesions

## DATASET

When we started this project, we did not have access to real pin site images. Therefore we used [ISIC archive](https://isic-archive.com/) images of melanomas (119) and not melanomas (87)

***Why this dataset?***

These lesions share similar properties to the images of pin site infections

## PROJECT LIFECYCLE 

### DATA PREPROCESSING 

Image Classification problems are highly complex and we know that…

***Complexity(Problem) ∝ Size(Data) ∝ Size(Model)***

This is a challenge in medical applications, as we normally have very limited number of images

How do you overcome the problem? 

***DATA AUGMENTATION***

Data augmentation implies increasing the amount of training data by applying transformations to both image and contour (so we could calculate the true bounding box). Below are the transformations applied:

0. Original Image

![Original Image](https://github.com/Martar189/Lession-Detection-with-YOLO/blob/master/images/10_NM6_orig.jpg)

1. Horizontal and vertical flips

![Flip1](https://github.com/Martar189/Lession-Detection-with-YOLO/blob/master/images/10_NM6_origflip_image.jpg)
![Flip2](https://github.com/Martar189/Lession-Detection-with-YOLO/blob/master/images/10_NM6_origfliphor_image.jpg)

2.	Rotations of 45 and 90 degrees

![Rotation1](https://github.com/Martar189/Lession-Detection-with-YOLO/blob/master/images/10_NM6_origrot45_image.jpg)
![Rotation2](https://github.com/Martar189/Lession-Detection-with-YOLO/blob/master/images/10_NM6_origrot90_image.jpg)

3.	Crop (30%)

![Crop](https://github.com/Martar189/Lession-Detection-with-YOLO/blob/master/images/10_NM6_origcrop_image.jpg)

4.	Scale up (1.5)

![Affine](https://github.com/Martar189/Lession-Detection-with-YOLO/blob/master/images/10_NM6_origaffine_image.jpg)

Initial set: 206 samples
Final dataset: Train on 1148 samples, validate on 42 samples

***Learnings***

We had to rebuild the pre - processing function to apply data augmentation to the training set only to avoid leakage
We needed to shuffle the data to avoid any inference based on the order of the images

### MODELLING

***OBJECT DETECTION USING YOLO (You Only Look Once)***

The original model structure includes included 23 CNN, with convolutional and max pooling layers, followed by 2 fully connected layers at the end. Uses a set of pre-trained weights that are then optimised using a custom loss function

![YOLO Architecture](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ43CZUbXypj0MJrgUP4d_PAlO8kwoKXL64B18rqamnz7r5B4bY)

![YOLO Custom Loss Function](https://github.com/Martar189/Lession-Detection-with-YOLO/blob/master/images/YOLO%20LOSS%20FUNCTION.jpg)


***Model advantages***

* The whole detection pipeline is a single network
* It can be customised and optimized end-to-end
* It is extremely fast

For more information on the model [click here](https://pjreddie.com/media/files/papers/yolo.pdf)

### OUR CUSTOM YOLO

A simplified version of YOLOv2 in Keras with Tensorlow backend using pre-trained weights

***What simplifications did we introduce?***

We were working under the assumption that our images only included one lession, hence we only needed to detect one bounding box

We broke the problem into tasks, allowing us to build the model iteratively. The first task being the bounding box detection, which allowes us to reduce our desired output was an array with only 4 elements

***What did this mean?***

1. From 23 to 22 layers with simple Dense output layer
2. From a complex custom loss function to mean squared error; treating this as a regression like problem

### HYPERPARAMETER TUNNING AND MODEL TRAINING

Hyperparameters are the model parameters whose value is initialized before the learning takes place

Hyperparamenter tunning refers to the process of determining the final value of those parameters ensuring the algorithm performs well not only on the training data, but also on new inputs

In neural networks there are many parameters to tune, below are the ones we explored in this project:

1. Transfer learning vs Fine tuning

Transfer Learning = transferring the weights of an already trained model to another problem, initialising the model with those trained weights and training the whole network on the new dataset, optimising all parameters

Fine Tuning = freezing the weights of all/some layers except the penultimate layer and train the network just to learn the representations of the penultimate layer

2. Learning Rate and Optimiser

3. Dropout Layer

Regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data. Neurons are cancelled at random based on the chosen droupout value

![Dropout explained](https://cdn-images-1.medium.com/max/600/0*dOZ6esAristenchF.png)









