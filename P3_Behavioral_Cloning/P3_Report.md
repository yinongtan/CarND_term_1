# **Project3: Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/convnet.png "Model Visualization"
[image2]: ./examples/center_lane.jpg "Grayscaling"
[image3]: ./examples/recovery_1.jpg "Recovery Image"
[image4]: ./examples/recovery_2.jpg "Recovery Image"
[image5]: ./examples/recovery_3.jpg "Recovery Image"
[image6]: ./examples/normal_image.jpg "Normal Image"
[image7]: ./examples/flipped_image.jpg "Flipped Image"
[image8]: ./examples/center_image.jpg "Center Image"
[image9]: ./examples/left_image.jpg "Left Image"
[image10]: ./examples/right_image.jpg "Right Image"
[image11]: ./examples/MSE_history.png "MSE_loss"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* P3_Report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 128 (model.py lines 75-85) 

The model includes RELU layers to introduce nonlinearity (code lines 75-79), and the data is normalized in the model using a Keras lambda layer (code line 73). The cameras in the simulator capture 160 pixel by 320 pixel images. However, not all of these pixels contain useful information. In the image above, the top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car. As a result, I also used a Keras Cropping2D layer (code line 74) to train the model faster by focusing on only the portion of the image that is useful for predicting a steering angle.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 81, 83). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 5-8). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a well known architecture used by NVIDIA to address the similar problem. Then adjust the architecture to eliminate underfitting and reduce overfitting for our dataset. Finally test the model using track 1 in the simulator to validate the performance.

My first step was to use a convolution neural network model similar to the [NVIDIA architecture](https://arxiv.org/pdf/1604.07316v1.pdf). I thought this model might be appropriate because this is the CNN acchitecture that used by NVIDIA to generate steering from the video images of a single center camera, which is very similiar to the project requirements.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the first fulled connected layer was removed. Reducing the number of neurons in the model can help us address overfitting.

Then I also add two dropout layers to the fully connected layers to further reduce the overfitting. The keep probability is set to be 0.5.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when the vehi. To improve the driving behavior in these cases, I augmented the dataset in two ways: 1. fipped the training images and took the opposite sign of the steering measurement (code line 48-53); 2. using data from multiple cameras and tuning the correction value to add and subtract from the center angle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Normalize         	| 160x320x3 normalized [-0.5, 0.5] image		| 
| Cropping2D         	| 65x320x3 normalized [-0.5, 0.5] image			| 
| Convolution1 5x5     	| 2x2 stride, valid padding, outputs 31x158x24 	|
| RELU					| Activation									|
| Convolution2 5x5     	| 2x2 stride, valid padding, outputs 14x77x36 	|
| RELU					| Activation									|
| Convolution3 5x5     	| 2x2 stride, valid padding, outputs 5x37x48 	|
| RELU					| Activation									|
| Convolution4 3x3     	| 1x1 stride, valid padding, outputs 3x35x64 	|
| RELU					| Activation									|
| Convolution5 3x3     	| 1x1 stride, valid padding, outputs 1x33x64 	|
| RELU					| Activation									|
| Flatten		      	| inputs 1x33x64,  outputs 2112					|
| Fully connected1		| inputs 2112, outputs 100        				|
| Dropout				| keep prob = 0.5								|
| Fully connected2		| inputs 100, outputs 50        				|
| Dropout				| keep prob = 0.5								|
| Fully connected3		| inputs 50, outputs 1	        				|


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it gets off to the side of the road. These images show what a recovery looks like starting from the right side of the road:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help with the left turn bias. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

In addition, I also o feed the left and right camera images to your model as if they were coming from the center camera. For example, here is the images that have been taken from all three cameras simultaneously:

![alt text][image8]
![alt text][image9]
![alt text][image10]

This way, I can teach the model how to steer if the car drifts off to the left or the right. After multiple experiments, the optimal correction value was found to be around 0.65.

After the collection process, I had 48216 number of data points. I then preprocessed this data by normalizing the images using a Keras lambda layer and cropping the top 70 and bottom 25 rows of pixels to focus on the most useful information of the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by the figure shown below:

![alt text][image11]

The mean squared error of the validation began to increase after 6 epoches. I used an adam optimizer so that manually training the learning rate wasn't necessary.
