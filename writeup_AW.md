# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./write_up_examples\example_each.jpg "Each class"
[image2]: ./write_up_examples\histogram_classes.jpg "Histogram"
[image3]: ./write_up_examples\grayscale_class.jpg  "Grayscale"

[image4]: ./Additional_test_images/add_image_1.jpg "Traffic Sign 1"
[image5]: ./Additional_test_images/add_image_2.jpg "Traffic Sign 2"
[image6]: ./Additional_test_images/add_image_3.jpg "Traffic Sign 3"
[image7]: ./Additional_test_images/add_image_4.png "Traffic Sign 4"
[image8]: ./Additional_test_images/add_image_5.png "Traffic Sign 5"

---

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The first image shows a random example of all 43 different types of traffic sign.

![alt text][image1]

The following diagram shows a histogram of the number of training examples of each class. There is significant variation but each one contains at least ~200 examples in the training set. 

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Data Preparation

Initially the only steps I completed was to normalise the data. This improves the networks ability to learn. 

Having tested a few passes of the network I found that I was getting very high variance. One way to deal with this would have been to increase the dataset. Instead I realised that most of the information about what a sign was is contained within the shapes of the sign I would convert to grayscale. This made a dramatic improvement. There are a number of ways to convert to grayscale, but a quick and computationally efficient way is to take an average of the 3 channels. Once I'd taken the average I needed to make sure that the shape was still correct.

```python
X_train = np.mean(X_train, axis=3)
X_train = np.reshape(X_train, (n_train,32,32,1))

```
Here is an example of the effects of greyscale.

![alt text][image3]

As I managed to reach the required accuracy I did not decide to increase the training set. To improve performance further this would likely have been my next step. It may have allowed me to train a model with color and not have such a high variance. With augumentation I would have to be careful to not use transformations (flip/rotation) that may turn one sign into another.


#### 2.  Model Architecture

The base model was based on LeNet. The main changes I made were to add in four dropout layers. This was because in my experimentation I was finding there a large variance between the train and validation datasets. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding						|
| RELU					|												|
| Dropout				| keep_prob = 0.6								|
| Max pooling	      	| 2x2 stride =2  								|
| Convolution 5x5     	| 1x1 stride, valid padding						|
| RELU					|												|
| Dropout				| keep_prob = 0.6								|
| Max pooling	      	| 2x2 stride =2  								|
| Fully connected		| 400-120     									|
| RELU					|												|
| Dropout				| keep_prob = 0.6								|
| Fully connected		| 120 -84     									|
| RELU					|												|
| Dropout				| keep_prob = 0.6								|
| Fully connected		| 84 - 43    									|


#### 3. Model Training

The following tests outline the different training steps I took. 

When running the output for the baseline model, by outputing the train accuracy, it's clear that the model is overfitting.For example- these are the baseline results after 10 epochs for the set up taken from the previous lessons. 

Validation Accuracy = 0.723
Train Accuracy = 0.958

The below discussion covers some of the changes I made to try and improve the validation accuracy. 

##### Test 1. Attempt to reduce overfitting
Added a dropout layer after the first and second convolution layer (after the relu) and after the first fully connected layer. 
Keep_prob = 0.5
lr = 0.001
batch size = 128

EPOCH 10 ...
Validation Accuracy = 0.630
Train Accuracy = 0.788

Still a lot of variance. Just seems to have brought both down. 

##### Test 2. Add another layer of dropout after second fully connected layer.
Keep_prob = 0.5
lr = 0.001
batch size = 128

EPOCH 10 ...
Validation Accuracy = 0.561
Train Accuracy = 0.661

This has made it considerably worse. The variance is now less but the bias has gone up. 
Looking at the trend it doesn't feel as though more epochs could help.


##### Test3. Reduce dropout to 0.6 keep and train for 15 epochs.
lr = 0.001
batch size = 128

EPOCH 15 ...
Validation Accuracy = 0.733
Train Accuracy = 0.872

It's getting better again but there's still a lot of variance. 

##### Test 4. Lower the learning rate by factor of 10 to 0.0001
lr = 0.0001
batch size = 128
EPOCH 25 ...
Validation Accuracy = 0.404
Train Accuracy = 0.544

This obviously takes a lot longer to train and so hasn't finished improving. 

EPOCH 100 ...
Validation Accuracy = 0.678
Train Accuracy = 0.848

Still a high variance. 


##### Test 5 Grayscale

Still a variance problem - maybe I need more data? Augmentation?
There isn't a great deal of color variation and the shape changes are the most important so worth testing greyscale.
lr = 0.001
batch size = 128
Test 5 is grayscale 

EPOCH 10 ...
Validation Accuracy = 0.929
Train Accuracy = 0.969

This made a massive difference. We're nearly at the required spec level. 


##### Test 6 Increase epochs
lr = 0.001
batch size = 128
To try and get over the 93% validation I'm going to increase the epochs to 20.

EPOCH 20 ...
Validation Accuracy = 0.941
Train Accuracy = 0.986

#### 4. Final Results

My final model results were:
* training set accuracy of 0.986
* validation set accuracy of 0.941
* test set accuracy of 0.906

The steps taken to achieve this is outlined in part 3.  

### Test a Model on New Images

#### 1. Additional web images
Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images are all clear with a simple white background and therefore I would expect them to be fairly easy to classify.

#### 2. Discussion on additional data

The accuracy on these additional images is only 40%. (2/5 are predicted correctly). This is interesting for a couple of reasons. The two that it got correct (speed limit and general caution) had 1260 and 1080 examples in the training data. The three that it got wrong had 0, 180 and 300 examples in the training set. 

This shows that using accuracy as a metric is not always the best thing to do. It may be hiding poor performance of classes with fewer examples. 

What is at least encouraging is that the ones it got wrong it had a very low confidence level for. 

Adding an additional road sign into this test that was not one of the 43 classes also shows the importance of having to consider how the model will manage with signs it has not seen before.

Here are the results of the prediction for the 5 additonal images:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)  | Speed limit (60km/h)							| 
| Two way road    		| Speed limit (20km/h) 							|
| General caution		| General caution								|
| Dangerous curve left	| Speed limit (60km/h)							|
| Roundabout mandatory	| Speed limit (60km/h)     						|

This is obviously too small a sample to make significant judgement on, but the learnings above suggest additional metrics and data augmentation of additonal data on the classes with few examples.


#### 3. Model Perfornance on Additional images.

![alt text][image4] 

-Pred 0, Speed limit (60km/h), probability 0.972
-Pred 1, Speed limit (80km/h), probability 0.0164
-Pred 2, Speed limit (50km/h), probability 0.0117
-Pred 3, Ahead only, probability 1.78e-05
-Pred 4, No passing for vehicles over 3.5 metric tons, probability 2.05e-06

![alt text][image5] 

-Pred 0, Speed limit (20km/h), probability 0.279
-Pred 1, Ahead only, probability 0.223
-Pred 2, Children crossing, probability 0.136
-Pred 3, Speed limit (60km/h), probability 0.127
-Pred 4, Go straight or right, probability 0.080

![alt text][image6] 

-Pred 0, General caution, probability 0.998
-Pred 1, Pedestrians, probability 0.0012
-Pred 2, Traffic signals, probability 0.000587
-Pred 3, Right-of-way at the next intersection, probability 4.077e-05
-Pred 4, Road narrows on the right, probability 9.278e-07

![alt text][image7] 

-Pred 0, Speed limit (60km/h), probability 0.218
-Pred 1, Speed limit (80km/h), probability 0.102
-Pred 2, Road work, probability 0.0893
-Pred 3, Children crossing, probability 0.0707
-Pred 4, Bicycles crossing, probability 0.0524

![alt text][image8]

-Pred 0, Speed limit (60km/h), probability 0.174
-Pred 1, Speed limit (80km/h), probability 0.126
-Pred 2, Road work, probability 0.0766
-Pred 3, Speed limit (120km/h), probability 0.0682
-Pred 4, Children crossing, probability 0.0449



