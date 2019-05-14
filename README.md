# Clock Project
This document will outline entirely how to reproduce our clock model in Google Colab.

# Step 1:
Obtain a set of images that shows every minute of every hour. 

This should give you a total of 720 images. This will allow us to get 12 outputs on our final dense layer. A variant to this would be using the image data generater to create additional photos so that you could do 720 outputs on your final dense layer

# Step2:
Import the necessary libraries.

The following libraries were used.
import keras
from keras.models import *
from keras.layers import *
import os, shutil
from keras.preprocessing import image
import matplotlib.pyplot as plt
from google.colab import files, drive
import pandas as pd
import tensorflow as tf

# Step 3:
Mount your Google Drive and then check what files are available by using the !ls command

be sure to use the full file path. 

A good example would be the following path 

"/content/drive/My Drive/Colab Notebooks/Clocks"

# Step 4:
set your directories up with your path to your train and test set

you will also need directories set up for your csv files.

open the file and read them

# Step 5:
define your functions for obtaining the labels, filenames and creating file paths.

we have 6 functions at the begginning of the notebook:
## getfiles():
we loop through the csv and split the lines by comma. We save the label twice, one goes to the getHour function and the other goes to annotateTime() or annotateTestTime().  This created '\n' so we ended up stripping that as well as '\ufeff' if it shows up on your data.
## getTrainAnalogFilepath():
this function fruns through the filenames which is created in the above function. The directory that holds the photos are then joined with the picture name and then appeneded.
## getTestFilepath():
This is exactly the same functionality as getTrainAnalogFilepath() but with training datasets directory path.
## annotateTime():
This is strictly for the training set as the dimensions are expanded to the number of photos we have for the training set. this runs through the time label and splits them time by ':' , we then set the values for the hour and for the minute to integers and if the hour is equal to 12 we set hour equal to zero. As we loop through the file we multiply the minute by 0.5 and had (30 * hour) to annotate the hour. To annotate te minute we multiply the minute by 6.
we then set the array equal to (100, 100, 1), then append this to itself on axis 0. The same thing is done with minutes. This gives you thetaH and thetaM. This should come out to a dimension of (720, 100, 100, 1) these will be used later.
## annotateTestTime():
The function above is similar but the array is only expanded to (100, 2)
## getHour():
This utilizes the second label we created in the first function. We loop through the file and then split the time by ':' , the item is then set to a temp variable and saved as an integer that has one subtracted from it and then appended to itself. The reason that we subtract one from each hour is so that the categorical number of classes equals out to twelve. If this is not done then the number of classes equals out to 13.

# Step 6:
get all of your labels, thetas and filepaths initialized and then verify the outputs are correct and that the shapes match up. 

# Step 7:
now you will set up your labels to categorical, make sure that your number of classes equals 12

# Step 8:
define a function loadImages()
this loops through the image paths with a target size of 100 by 100 and grayscaled. The image gets loaded to an array, the dimensions are expanded on the zero axis. Divide this by 255.0 and append this to to an array. After this is completed concatenate everything on the zero axis. This may take a bit even with the GPU running. Do a quick plt of a few photos as a sanity check.

# Step 9:
concat the images and theta hour and minute on the 4th axis. If done correctly on the training images the dimensions come out to (720, 100, 100, 3). If done correctly on the test set the dimensions will be (100, 100, 100, 3)

# Step 10:
next you will need to run from sklearn.utils import shuffle
shuffle your images and outputs together

# Step 11:
next run your dataset and outputs through K-Fold

decide how you want to divide your train and validation sets. We have ours set to 4 iterations which gives us a total train set of 540 pictures and 180 validation pictures. I like to print out the shapes to validate any changes made if we increase or decrease the number of iterations.

# Step 12:
next you will want to define your residual network and build your model. We decided to add call backs to save at the best checkpoint and send a copy of our saved model to Google Drive. We also implemented early stopping with a threshold of 100 epochs.

# Step 13:
Plot the accuracy and loss for the training and validation over the number of epochs.

# Step 14:
After you have built a model with the best validation accuracy possible load this model 

# Step 15:
Shuffle your test images and outputs and evaluate on the loaded model.

run a few predictions and see how everything goes.
