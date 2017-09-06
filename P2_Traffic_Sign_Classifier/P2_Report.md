# **Project2: Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./num_each_label_train.png "Visualization_train"
[image2]: ./num_each_label_valid.png "Visualization_valid"
[image3]: ./num_each_label_test.png "Visualization_test"
[image4]: ./grayscale.png "grayscale"
[image5]: ./normalization.png "normalization"
[image6]: ./test_images.png "test_images"
[image7]: ./test_images_with_pred.png "test_images_with_pred"
[image8]: ./top_5_0.png "top_5_0"
[image9]: ./top_5_1.png "top_5_1"
[image10]: ./top_5_2.png "top_5_2"
[image11]: ./top_5_3.png "top_5_3"
[image12]: ./top_5_4.png "top_5_4"
[image13]: ./top_5_5.png "top_5_5"
[image14]: ./top_5_6.png "top_5_6"
[image15]: ./top_5_7.png "top_5_7"
[image16]: ./top_5_8.png "top_5_8"
[image17]: ./top_5_9.png "top_5_9"





<!-- [image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5" -->

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yinongtan/CarND_term_1/blob/master/P2_Traffic_Sign_Classifier/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799. 
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is [32x32x3].
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data distributes among all the 43 classes.

![alt text][image1]

Similarly, two charts summarizing the validation dataset and the test dataset are shown below as well.
![alt text][image2]

![alt text][image3]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the classes of traffic signs mainly depend on their shapes, not colors. Introducing three RGB channels of colors will not increase the accuracy of the traffic sign classification. In addition, it will make the CNN calculation slower and increase noises. Consequently, I decided to convert the RGB images to grayscaled images first.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

As a last step, I normalized the image data so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and was used in this project. is In this case, all the weights and biases for the neural network will also be kept in similar scaling.

Here is an example of a traffic sign image before and after normalization.

![alt text][image5]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Grayscale        		| 32x32x1 grayscale image   					| 
| Normalize         	| 32x32x1 normalized [-1, 1] image				| 
| Convolution1 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					| Activation									|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12					|
| Convolution2 5x5	    | 1x1 stride, valid padding, outputs 10x10x32 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32					|
| Flatten		      	| inputs 5x5x32,  outputs 800					|
| Fully connected1		| inputs 800, outputs 240        				|
| Dropout				| keep prob = 0.5								|
| RELU					| Activation									|
| Fully connected2		| inputs 240, outputs 168        				|
| Dropout				| keep prob = 0.5								|
| RELU					| Activation									|
| Fully connected3		| inputs 168, outputs 43        				|
| Softmax				| Computes softmax cross entropy         		|
| L2 Regularization		| beta = 0.003         							|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with a batch size of 100 and a epochs number of 10. The learning rate is set to be 0.001 and the beta value for L2 regularization is 0.003.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.0%
* validation set accuracy of 95.1%
* test set accuracy of 93.2%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture that was tried was the original LeNet architecture. The reason is that both the traffic sign classifier and LeNet focus on classifying images into several given labels based on all the important features shown in the images.
* What were some problems with the initial architecture?
The main problem with the initial architecture is that the prediction accuracy of the training set and the validation set are low (around 89%), which implies underfitting.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

First we need to eliminate underfitting. Because the traffic sign data has more labels (43 classes compared to 10 classes in LeNet) and also more features, the depth of the convolutional layers and the unit number of FC layer are both set to be twice larger in order is set to be twice larger to capture more features in the images. 

After running the simulation, I found that although this modified architecture works pretty well for the training set. However, the accuracy of the validation set is still relatively low. This indictes overfitting. 

To solve the problem of overfitting, I tried two methods. 1. Using dropout for all 3 FC layers with a keep probability of 0.5; 2. Using L2 regularization to penalize all the weights in all convolutional layers and FC layers.

* Which parameters were tuned? How were they adjusted and why?

The dropout keep probability is tuned from 1, 0.8, 0.5, 0.3. The results showed that if the probability is too high then dropout cannot reduce overfitting. On the other hand, if the probability is too low then it will cause underfitting. Among these four values, 0.5 is the optimal value.

The beta value of the L2 regularization is tuned from 0.001, 0.003, 0.005, 0.01. The results showed that if the beta value is too low then dropout cannot reduce overfitting. On the other hand, if the beta value is too high then it will cause underfitting. So beta's influence on the accuracy is just the opposite to the dropout keep probability. Among these four values, 0.003 is the optimal value.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

A convolution layer is the most important design choice in this problem. Because the meaning of a traffic sign should stay the same regardless of the position it appears in the image or the angle it rotates. The dropout method and L2 regularization are also very important because they can help us reduce the overfitting problem.


If a well known architecture was chosen:
* What architecture was chosen?

LeNet was chosen as the starting point of my artitecture.

* Why did you believe it would be relevant to the traffic sign application?

Because both the traffic sign classifier and LeNet focus on classifying images into several given labels based on all the important features shown in the images. Specially, LeNet contains convolution, pooling, fully connected layers, which are essential for building a powerful CNN. So I considered LeNet a very good starting point for this problem.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The final model provides a training set accuracy of 99.0%, a validation set accuracy of 95.1%, a test set accuracy of 93.8%. Since it is not possible to totally eliminate the influence of overfitting, it is acceptable that we have managed to increase the accuracy of validation set to 95.1%. Even the test dataset has a accuracy greater than the threshold (93%). In conclusion, the model is working well to classify the German traffic signs.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image6] 


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image7] 

| Image			        				 	| Prediction	        					| 
|:-----------------------------------------:|:-----------------------------------------:| 
| Vehicle over 3.5 metric tons prohibited	| Vehicle over 3.5 metric tons prohibited	| 
| Speed limit (30km/h)     					| Speed limit (30km/h) 						|
| Keep right								| Keep right								|
| Turn right ahead	      					| Turn right ahead					 		|
| Right-of-way at the next intersection		| Right-of-way at the next intersection     |
| Keep right      							| Keep right   								| 
| Generation caution     					| Generation caution 						|
| Priority road								| Priority road								|
| Road work						      		| Road work					 				|
| Ahead only								| Ahead only      							|



The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.8%. Because all these ten test images are very bright images with little distortion, the prediction results on these new test images are very accurate.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Vehicle over 3.5 metric tons prohibited sign (probability of 1), and the image does contain a Vehicle over 3.5 metric tons prohibited sign. The top five soft max probabilities were

![alt text][image8] 
<!-- | Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							| -->

For the second image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 1), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

![alt text][image9]

For the third image, the model is relatively sure that this is a Keep right sign (probability of 1), and the image does contain a Keep right sign. The top five soft max probabilities were

![alt text][image10]

For the fourth image, the model is relatively sure that this is a Turn right ahead sign (probability of 1), and the image does contain a Turn right ahead sign. The top five soft max probabilities were

![alt text][image11]

For the fifth image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 1), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

![alt text][image12]

For the sixth image, the model is relatively sure that this is a Keep right sign (probability of 1), and the image does contain a Keep right sign. The top five soft max probabilities were

![alt text][image13]

For the seventh image, the model is relatively sure that this is a Generation caution sign (probability of 1), and the image does contain a Generation caution sign. The top five soft max probabilities were

![alt text][image14]

For the eighth image, the model is relatively sure that this is a Priority road sign (probability of 1), and the image does contain a Priority road sign. The top five soft max probabilities were

![alt text][image15]

For the ninth image, the model is relatively sure that this is a Road work sign (probability of 1), and the image does contain a Road work sign. The top five soft max probabilities were

![alt text][image16]

For the tenth image, the model is relatively sure that this is a Ahead only sign (probability of 1), and the image does contain a Ahead only sign. The top five soft max probabilities were

![alt text][image17]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


