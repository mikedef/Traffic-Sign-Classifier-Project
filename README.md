# Traffic-Sign-Classifier-Project

## Michael DeFilippo

### This project was used to introduce the concepts of using Tensorflow and the Python programming language to build a German traffic sign classifier based on a LeNet-5 convolutional network (CovNet). Once the CovNet was trained on a data set of German traffic signs I was able to test this on a validation dataset and a test dataset to test my classifiers accuracy. Furthermore I tested this CovNet on six randomly selected German traffic signs that I downloaded from Google Images. 

#### Please see my [project code](https://github.com/mikedef/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) for any questions regarding implimentation. 
---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set

I used the numpy and pandas library to calculate summary statistics of the traffic signs data set:

    The size of training set is 34799 images. 
    The size of the validation set is 4410 images.
    The size of test set is 12630 images.
    The shape of a traffic sign image is (32, 32).
    The number of unique classes/labels in the data set is 43.

I was able to also view the sign labels after reading in the signnames.csv with the pandas library. This was useful as I was able to see what each class name was supposed to be labeled as. 

#### 2. Include an exploratory visualization of the dataset

Here is an exploratory visualization of the data set. It is a bar chart showing how the images are distributed across the 43 classes. 

![png][writeup/ClassHist.png]
