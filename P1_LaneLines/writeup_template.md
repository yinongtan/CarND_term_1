# **Project: Finding Lane Lines on the Road** 

<!-- ## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report -->


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[step1_solidWhiteCurve]: ./test_images_output/step1_solidWhiteCurve.jpg
[step1_solidWhiteRight]: ./test_images_output/step1_solidWhiteRight.jpg
[step1_solidYellowCurve]: ./test_images_output/step1_solidYellowCurve.jpg
[step1_solidYellowCurve2]: ./test_images_output/step1_solidYellowCurve2.jpg
[step1_solidYellowLeft]: ./test_images_output/step1_solidYellowLeft.jpg
[step1_whiteCarLaneSwitch]: ./test_images_output/step1_whiteCarLaneSwitch.jpg

[step2_solidWhiteCurve]: ./test_images_output/step2_solidWhiteCurve.jpg
[step2_solidWhiteRight]: ./test_images_output/step2_solidWhiteRight.jpg
[step2_solidYellowCurve]: ./test_images_output/step2_solidYellowCurve.jpg
[step2_solidYellowCurve2]: ./test_images_output/step2_solidYellowCurve2.jpg
[step2_solidYellowLeft]: ./test_images_output/step2_solidYellowLeft.jpg
[step2_whiteCarLaneSwitch]: ./test_images_output/step2_whiteCarLaneSwitch.jpg

[step3_solidWhiteCurve]: ./test_images_output/step3_solidWhiteCurve.jpg
[step3_solidWhiteRight]: ./test_images_output/step3_solidWhiteRight.jpg
[step3_solidYellowCurve]: ./test_images_output/step3_solidYellowCurve.jpg
[step3_solidYellowCurve2]: ./test_images_output/step3_solidYellowCurve2.jpg
[step3_solidYellowLeft]: ./test_images_output/step3_solidYellowLeft.jpg
[step3_whiteCarLaneSwitch]: ./test_images_output/step3_whiteCarLaneSwitch.jpg

[step4_solidWhiteCurve]: ./test_images_output/step4_solidWhiteCurve.jpg
[step4_solidWhiteRight]: ./test_images_output/step4_solidWhiteRight.jpg
[step4_solidYellowCurve]: ./test_images_output/step4_solidYellowCurve.jpg
[step4_solidYellowCurve2]: ./test_images_output/step4_solidYellowCurve2.jpg
[step4_solidYellowLeft]: ./test_images_output/step4_solidYellowLeft.jpg
[step4_whiteCarLaneSwitch]: ./test_images_output/step4_whiteCarLaneSwitch.jpg

[step5_solidWhiteCurve]: ./test_images_output/solidWhiteCurve.jpg
[step5_solidWhiteRight]: ./test_images_output/solidWhiteRight.jpg
[step5_solidYellowCurve]: ./test_images_output/solidYellowCurve.jpg
[step5_solidYellowCurve2]: ./test_images_output/solidYellowCurve2.jpg
[step5_solidYellowLeft]: ./test_images_output/solidYellowLeft.jpg
[step5_whiteCarLaneSwitch]: ./test_images_output/whiteCarLaneSwitch.jpg


---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I used Gaussian smoothing to suppress noise and spurious gradients by averaging. The results after step 1 are shown as follows:

![alt text][step1_solidWhiteCurve]
![alt text][step1_solidWhiteRight]
![alt text][step1_solidYellowCurve]
![alt text][step1_solidYellowCurve2]
![alt text][step1_solidYellowLeft]
![alt text][step1_whiteCarLaneSwitch]

For step 2, I defined low threshold and high threshold for Canny edge detection to be 50 and 150 repectively, and then I applied Canny edge detection for the smoothed images. The results after step 2 are shown as follows:

![alt text][step2_solidWhiteCurve]
![alt text][step2_solidWhiteRight]
![alt text][step2_solidYellowCurve]
![alt text][step2_solidYellowCurve2]
![alt text][step2_solidYellowLeft]
![alt text][step2_whiteCarLaneSwitch]

For step 3, I used a four sided polygon to define the region of interest of each image. THe results after step 3 are shown as follows:

![alt text][step3_solidWhiteCurve]
![alt text][step3_solidWhiteRight]
![alt text][step3_solidYellowCurve]
![alt text][step3_solidYellowCurve2]
![alt text][step3_solidYellowLeft]
![alt text][step3_whiteCarLaneSwitch]

For step 4, I defined the Hough transform parameters (\rho = 2, \theta = 1 deg, threshold = 50, min line length = 25, max line gap = 25), then I applied Hough transform on edge detected images. The results after step 4 are shown as follows:

![alt text][step4_solidWhiteCurve]
![alt text][step4_solidWhiteRight]
![alt text][step4_solidYellowCurve]
![alt text][step4_solidYellowCurve2]
![alt text][step4_solidYellowLeft]
![alt text][step4_whiteCarLaneSwitch]

For step 5, I use the add weighted function in OpenCV to draw the lines on the edge image and make the lines semi transparent. The results after step 5 are shown as follows:

![alt text][step5_solidWhiteCurve]
![alt text][step5_solidWhiteRight]
![alt text][step5_solidYellowCurve]
![alt text][step5_solidYellowCurve2]
![alt text][step5_solidYellowLeft]
![alt text][step5_whiteCarLaneSwitch]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first initializing four lists to store the x and y coordinates for the left and right line respectively. Next, I traversed each line segments to decide whether it belongs to the left line or the right line. 

The criteria I set for the line segments classification is: if the slope is positive and the midpoint is on the right half side of the image, then this line segment is part of the right line; if the slope is negative and the midpoint is on the left half side of the image, then this line segment is part of the left line. 

Next, I used the polynomial fit function: np.polyfit (with degree 1 for straight line) to find the coefficients for the left and right line. Because the y coordinates for left and right lane lines are the same as the y coordinates of the region of interest. I used the function: np.poly1d and the y coordinates to find the x coordinates for the left and right lane lines. Finally, these points are used as the end points for my lane lines.


### 2. Identify potential shortcomings with your current pipeline

There are two main shortcomings of this pipeline, which are also the main reasons why the detection failed with the challenge video.

One potential shortcoming would be that when the vehicle goes into a road curve, the polynomial fitting becomes inaccurate. Because for road curves, each line segment has different slopes. The polynomial fitting would not work well if all the line segments in the region of interest are used as the training data to generate the straight line.

Another shortcoming could be for lines with yellow color, the detection works not as good as it for white lines. This is because the contrast between the solid yellow line on the left and the road surface is not large enough to trigger a Canny edge using the same pipeline as the white lines. Besides, the shadows in the video also introduce a lot of noise to the detection.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use a buffer to store itthe last slope values for left and right lane lines. And then use the weighted combination of the previous and current value to calcualte the final slope values for lane lines. This approach can avoid rapid changes of the slope and make the detection results more stable. 

Another potential improvement could be to to apply the HSV (hue, saturation, value) colorspace to boost the yellow areas of the image. This method can help us to distinguish the yellow line from the road surface better and reduce the noise in the video, thus improving the second shortcoming.
