## Table of Contents

- [Packages](#0)
- [1 - Problem Statement](#1)
- [2 - YOLO](#2)
    - [2.1 - Model Details](#2-1)
    - [2.2 - Filtering with a Threshold on Class Scores](#2-2)
        - [Exercise 1 - yolo_filter_boxes](#ex-1)
    - [2.3 - Non-max Suppression](#2-3)
        - [Exercise 2 - iou](#ex-2)
    - [2.4 - YOLO Non-max Suppression](#2-4)
        - [Exercise 3 - yolo_non_max_suppression](#ex-3)
    - [2.5 - Wrapping Up the Filtering](#2-5)
        - [Exercise 4 - yolo_eval](#ex-4)
- [3 - Test YOLO Pre-trained Model on Images](#3)
    - [3.1 - Defining Classes, Anchors and Image Shape](#3-1)
    - [3.2 - Loading a Pre-trained Model](#3-2)
    - [3.3 - Convert Output of the Model to Usable Bounding Box Tensors](#3-3)
    - [3.4 - Filtering Boxes](#3-4)
    - [3.5 - Run the YOLO on an Image](#3-5)
- [4 - Summary for YOLO](#4)
- [5 - References](#5)

<a name='0'></a>
## Packages

Run the following cell to load the packages and dependencies that will come in handy as you build the object detector!


```python
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image

%matplotlib inline
```

<a name='1'></a>
## 1 - Problem Statement

You are working on a self-driving car. Go you! As a critical component of this project, you'd like to first build a car detection system. To collect data, you've mounted a camera to the hood (meaning the front) of the car, which takes pictures of the road ahead every few seconds as you drive around. 

<center>
<video width="400" height="200" src="nb_images/road_video_compressed2.mp4" type="video/mp4" controls>
</video>
</center>

<caption><center> Pictures taken from a car-mounted camera while driving around Silicon Valley. <br> Dataset provided by <a href="https://www.drive.ai/">drive.ai</a>.
</center></caption>

You've gathered all these images into a folder and labelled them by drawing bounding boxes around every car you found. Here's an example of what your bounding boxes look like:

<img src="nb_images/box_label.png" style="width:500px;height:250;">
<caption><center> <u><b>Figure 1</u></b>: Definition of a box<br> </center></caption>

If there are 80 classes you want the object detector to recognize, you can represent the class label $c$ either as an integer from 1 to 80, or as an 80-dimensional vector (with 80 numbers) one component of which is 1, and the rest of which are 0. The video lectures used the latter representation; in this notebook, you'll use both representations, depending on which is more convenient for a particular step.  

In this exercise, you'll discover how YOLO ("You Only Look Once") performs object detection, and then apply it to car detection. Because the YOLO model is very computationally expensive to train, the pre-trained weights are already loaded for you to use. 

<a name='2'></a>
## 2 - YOLO

"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

<a name='2-1'></a>
### 2.1 - Model Details

#### Inputs and outputs
- The **input** is a batch of images, and each image has the shape (m, 608, 608, 3)
- The **output** is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers $(p_c, b_x, b_y, b_h, b_w, c)$ as explained above. If you expand $c$ into an 80-dimensional vector, each bounding box is then represented by 85 numbers. 

#### Anchor Boxes
* Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes.  For this assignment, 5 anchor boxes were chosen for you (to cover the 80 classes), and stored in the file './model_data/yolo_anchors.txt'
* The dimension for anchor boxes is the second to last dimension in the encoding: $(m, n_H,n_W,anchors,classes)$.
* The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).  


#### Encoding
Let's look in greater detail at what this encoding represents. 

<img src="nb_images/architecture.png" style="width:700px;height:400;">
<caption><center> <u><b> Figure 2 </u></b>: Encoding architecture for YOLO<br> </center></caption>

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Since you're using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.

For simplicity, you'll flatten the last two dimensions of the shape (19, 19, 5, 85) encoding, so the output of the Deep CNN is (19, 19, 425).

<img src="nb_images/flatten.png" style="width:700px;height:400;">
<caption><center> <u><b> Figure 3 </u></b>: Flattening the last two last dimensions<br> </center></caption>

#### Class score

Now, for each box (of each cell) you'll compute the following element-wise product and extract a probability that the box contains a certain class.  
The class score is $score_{c,i} = p_{c} \times c_{i}$: the probability that there is an object $p_{c}$ times the probability that the object is a certain class $c_{i}$.

<img src="nb_images/probability_extraction.png" style="width:700px;height:400;">
<caption><center> <u><b>Figure 4</u></b>: Find the class detected by each box<br> </center></caption>

##### Example of figure 4
* In figure 4, let's say for box 1 (cell 1), the probability that an object exists is $p_{1}=0.60$.  So there's a 60% chance that an object exists in box 1 (cell 1).  
* The probability that the object is the class "category 3 (a car)" is $c_{3}=0.73$.  
* The score for box 1 and for category "3" is $score_{1,3}=0.60 \times 0.73 = 0.44$.  
* Let's say you calculate the score for all 80 classes in box 1, and find that the score for the car class (class 3) is the maximum.  So you'll assign the score 0.44 and class "3" to this box "1".

#### Visualizing classes
Here's one way to visualize what YOLO is predicting on an image:

- For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across the 80 classes, one maximum for each of the 5 anchor boxes).
- Color that grid cell according to what object that grid cell considers the most likely.

Doing this results in this picture: 

<img src="nb_images/proba_map.png" style="width:300px;height:300;">
<caption><center> <u><b>Figure 5</u></b>: Each one of the 19x19 grid cells is colored according to which class has the largest predicted probability in that cell.<br> </center></caption>

Note that this visualization isn't a core part of the YOLO algorithm itself for making predictions; it's just a nice way of visualizing an intermediate result of the algorithm. 

#### Visualizing bounding boxes
Another way to visualize YOLO's output is to plot the bounding boxes that it outputs. Doing that results in a visualization like this:  

<img src="nb_images/anchor_map.png" style="width:200px;height:200;">
<caption><center> <u><b>Figure 6</u></b>: Each cell gives you 5 boxes. In total, the model predicts: 19x19x5 = 1805 boxes just by looking once at the image (one forward pass through the network)! Different colors denote different classes. <br> </center></caption>

#### Non-Max suppression
In the figure above, the only boxes plotted are ones for which the model had assigned a high probability, but this is still too many boxes. You'd like to reduce the algorithm's output to a much smaller number of detected objects.  

To do so, you'll use **non-max suppression**. Specifically, you'll carry out these steps: 
- Get rid of boxes with a low score. Meaning, the box is not very confident about detecting a class, either due to the low probability of any object, or low probability of this particular class.
- Select only one box when several boxes overlap with each other and detect the same object.

<a name='2-2'></a>
### 2.2 - Filtering with a Threshold on Class Scores

You're going to first apply a filter by thresholding, meaning you'll get rid of any box for which the class "score" is less than a chosen threshold. 

The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It's convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:  
- `box_confidence`: tensor of shape $(19, 19, 5, 1)$ containing $p_c$ (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
- `boxes`: tensor of shape $(19, 19, 5, 4)$ containing the midpoint and dimensions $(b_x, b_y, b_h, b_w)$ for each of the 5 boxes in each cell.
- `box_class_probs`: tensor of shape $(19, 19, 5, 80)$ containing the "class probabilities" $(c_1, c_2, ... c_{80})$ for each of the 80 classes for each of the 5 boxes per cell.

<a name='ex-1'></a>
### Exercise 1 - yolo_filter_boxes

Implement `yolo_filter_boxes()`.
1. Compute box scores by doing the elementwise product as described in Figure 4 ($p \times c$).  
The following code may help you choose the right operator: 
```python
a = np.random.randn(19, 19, 5, 1)
b = np.random.randn(19, 19, 5, 80)
c = a * b # shape of c will be (19, 19, 5, 80)
```
This is an example of **broadcasting** (multiplying vectors of different sizes).

2. For each box, find:
    - the index of the class with the maximum box score
    - the corresponding box score
    
    **Useful References**
        * [tf.math.argmax](https://www.tensorflow.org/api_docs/python/tf/math/argmax)
        * [tf.math.reduce_max](https://www.tensorflow.org/api_docs/python/tf/math/reduce_max)

    **Helpful Hints**
        * For the `axis` parameter of `argmax` and `reduce_max`, if you want to select the **last** axis, one way to do so is to set `axis=-1`.  This is similar to Python array indexing, where you can select the last position of an array using `arrayname[-1]`.
        * Applying `reduce_max` normally collapses the axis for which the maximum is applied.  `keepdims=False` is the default option, and allows that dimension to be removed.  You don't need to keep the last dimension after applying the maximum here.


3. Create a mask by using a threshold. As a reminder: `([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4)` returns: `[False, True, False, False, True]`. The mask should be `True` for the boxes you want to keep. 

4. Use TensorFlow to apply the mask to `box_class_scores`, `boxes` and `box_classes` to filter out the boxes you don't want. You should be left with just the subset of boxes you want to keep.   

    **One more useful reference**:
    * [tf.boolean mask](https://www.tensorflow.org/api_docs/python/tf/boolean_mask)  

   **And one more helpful hint**: :) 
    * For the `tf.boolean_mask`, you can keep the default `axis=None`.


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: yolo_filter_boxes

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
        boxes -- tensor of shape (19, 19, 5, 4)
        box_confidence -- tensor of shape (19, 19, 5, 1)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    ### START CODE HERE
    # Step 1: Compute box scores
    ##(≈ 1 line)
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    ##(≈ 2 lines)
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)
    
    print("-" * 30)
    print(box_scores.shape)
    print(box_classes.shape)
    print(box_class_scores.shape)
    print("-" * 30)
    print(box_scores)
    print("-" * 10)
    print(box_classes)
    print("-" * 10)
    print(box_class_scores)
    print("-" * 30)
    '''
    ------------------------------
    (19, 19, 5, 80)
    (19, 19, 5)
    (19, 19, 5)
    ------------------------------
    '''
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ## (≈ 1 line)
    filtering_mask = box_class_scores >= threshold
    
    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    ## (≈ 3 lines)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes , filtering_mask)
    ### END CODE HERE
    print("*" * 30)
    print(scores.shape)
    print(boxes.shape)
    print(classes.shape)
    print("-" * 30)
    print(scores)
    print("-" * 10)
    print(boxes)
    print("-" * 10)
    print(classes)
    print("-" * 30)
    
    return scores, boxes, classes
```


```python
# BEGIN UNIT TEST
tf.random.set_seed(10)
box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = 0.5)
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))

assert type(scores) == EagerTensor, "Use tensorflow functions"
assert type(boxes) == EagerTensor, "Use tensorflow functions"
assert type(classes) == EagerTensor, "Use tensorflow functions"

assert scores.shape == (1789,), "Wrong shape in scores"
assert boxes.shape == (1789, 4), "Wrong shape in boxes"
assert classes.shape == (1789,), "Wrong shape in classes"

assert np.isclose(scores[2].numpy(), 9.270486), "Values are wrong on scores"
assert np.allclose(boxes[2].numpy(), [4.6399336, 3.2303846, 4.431282, -2.202031]), "Values are wrong on boxes"
assert classes[2].numpy() == 8, "Values are wrong on classes"

print("\033[92m All tests passed!")
# END UNIT TEST
```

    ------------------------------
    (19, 19, 5, 80)
    (19, 19, 5)
    (19, 19, 5)
    ------------------------------
    tf.Tensor(
    [[[[ 4.28564978e+00  6.49076509e+00 -6.30081117e-01 ...  6.68808997e-01
        -1.68395257e+00  1.03826976e+00]
       [ 4.94368935e+00  4.77268457e+00  5.33019590e+00 ...  5.31939554e+00
        -2.80533719e+00 -8.73986053e+00]
       [-6.94701910e+00  6.49599284e-02 -9.23110694e-02 ... -9.84258413e-01
         3.70492244e+00  1.84178233e+00]
       [ 3.84039074e-01  8.50888157e+00  3.17113352e+00 ... -7.25377893e+00
        -8.73767090e+00 -9.43146992e+00]
       [-7.93190813e+00  1.33746588e+00  2.40326667e+00 ...  2.39685392e+00
         2.11264062e+00  2.00848174e+00]]
    
      [[ 1.00695286e+01 -7.28247881e+00  4.87189102e+01 ... -2.98731613e+00
        -2.32418900e+01 -2.69899178e+01]
       [-1.64068714e-01  1.30940080e+00  1.98728001e+00 ... -4.02586460e+00
        -3.73160052e+00 -1.10689104e+00]
       [ 1.27394661e-01 -1.31044221e+00 -1.67494261e+00 ... -5.20223975e-01
        -7.39511132e-01  1.21109605e+00]
       [-3.69557232e-01  3.91533583e-01 -7.72597122e+00 ...  1.53176174e+01
         9.69397354e+00  5.94132066e-01]
       [-2.25000501e+00  1.58550777e+01  9.32684135e+00 ... -6.16618013e+00
         2.78169289e+01  1.82772331e+01]]
    
      [[ 9.13469374e-01  5.15317297e+00 -1.93471682e+00 ... -4.37535137e-01
        -5.86104488e+00  3.92505670e+00]
       [ 1.89848900e+01  2.54602776e+01  2.14270954e+01 ... -2.67497969e+00
         3.71488419e+01  1.56921082e+01]
       [ 6.50590897e+00 -1.01608295e+01  3.17006564e+00 ... -5.32510817e-01
         1.46294403e+01  1.03398490e+00]
       [-2.87513256e+00  2.68601532e+01  9.04059315e+00 ... -1.29549084e+01
        -7.76617861e+00  2.61932068e+01]
       [ 4.67850685e+00 -7.81025028e+00 -8.75759125e+00 ...  1.02901483e+00
        -8.27609158e+00  5.17865801e+00]]
    
      ...
    
      [[-1.99243832e+01  1.68817196e+01  6.24090652e+01 ...  4.15034523e+01
         3.27782440e+01  3.90553665e+01]
       [-2.32876549e+01  1.70992508e+01  1.61496563e+01 ...  6.74947453e+00
         2.89175510e+00  1.01255207e+01]
       [ 4.80032682e+00 -2.28062463e+00 -1.20319185e+01 ... -6.19316006e+00
        -4.17873621e+00 -5.25822592e+00]
       [-3.50430036e+00 -2.95889091e+01  1.18113804e+01 ... -1.57953465e+00
         5.01144753e+01 -3.33911171e+01]
       [ 1.19604006e+01 -9.16006851e+00 -3.82854424e+01 ... -4.54286480e+00
         1.49582577e+01 -6.14043570e+00]]
    
      [[-1.20762959e+01  1.87225986e+00 -3.05408764e+00 ... -6.78894377e+00
         5.38638878e+00 -1.40304193e+01]
       [ 1.89294453e+01  2.75461140e+01  9.29965115e+00 ... -1.72064285e+01
         2.43684387e+00  1.83169594e+01]
       [-4.49017601e+01  3.57998619e+01  1.90651000e+00 ... -4.56773415e+01
         7.45132399e+00  2.77008877e+01]
       [ 1.05153332e+01  1.50519335e+00 -4.38341570e+00 ...  1.73546267e+00
        -3.49746060e+00 -9.75717831e+00]
       [-9.60766602e+00  4.61507893e+00 -6.46754360e+00 ...  2.06437073e+01
         5.77789068e-01 -2.91008806e+00]]
    
      [[ 2.32391891e+01  2.86166906e+00 -2.52889519e+01 ...  9.42857170e+00
         5.36092043e-01 -1.27840443e+01]
       [-3.93612802e-01  1.62006970e-02 -2.64839884e-02 ... -4.62732840e+00
        -1.70472658e+00 -1.03727794e+00]
       [-1.65924854e+01 -7.32946968e+00 -1.91738868e+00 ...  2.81849155e+01
        -2.34573956e+01 -1.34706717e+01]
       [-5.05391998e+01 -2.52074585e+01 -1.86510372e+01 ...  3.97013397e+01
        -6.48946915e+01  2.72658997e+01]
       [-4.48057222e+00  1.74620800e+01 -1.64522672e+00 ... -3.09696960e+00
        -6.21778393e+00  1.54195833e+01]]]
    
    
     [[[-1.85061016e+01  4.94595194e+00  1.32036562e+01 ...  1.34213448e+00
         9.68435097e+00  5.98644733e+00]
       [-6.34147167e+00  2.65377569e+00 -4.47947073e+00 ...  5.67317390e+00
        -1.84978759e+00 -1.26310272e+01]
       [-4.65793839e+01  5.47178078e+01  4.94640198e+01 ...  9.44636762e-02
         3.53295250e+01 -1.41693048e+01]
       [-2.11872911e+00 -1.59458828e+00 -2.26306796e+00 ...  1.24064374e+00
        -5.70949984e+00  3.07475775e-01]
       [ 1.26948414e+01  4.09376183e+01  9.17198753e+00 ...  4.56927109e+00
        -8.90168095e+00  2.71561222e+01]]
    
      [[-4.76142690e-02 -9.87521076e+00  1.51656830e+00 ... -9.54270244e-01
         2.99165058e+00  7.90200853e+00]
       [ 1.07254047e+01 -2.18886528e+01  1.10443563e+01 ... -2.76554947e+01
         7.05519180e+01 -1.85728798e+01]
       [-4.22855902e+00 -8.28805506e-01 -1.97250450e+00 ...  5.93443823e+00
         1.21887517e+00  5.83469391e-01]
       [ 4.27113581e+00 -3.57382202e+00 -6.61766243e+00 ...  2.04191818e+01
         2.12772579e+01  3.32596016e+00]
       [ 1.30488691e+01  7.43278885e+00 -2.22990856e+01 ... -1.26257944e+01
        -2.13545094e+01 -5.00429821e+00]]
    
      [[ 2.07457447e+01 -6.53777084e+01  3.76241837e+01 ...  4.19900894e+01
         7.65826035e+00 -3.19824696e+00]
       [ 7.67650414e+00 -3.86734056e+00  7.64900637e+00 ... -7.16125250e+00
         1.34598579e+01 -7.82284451e+00]
       [ 2.63608408e+00  1.02151024e+00 -5.58251286e+00 ... -3.92365694e+00
        -3.04345918e+00  8.68897438e-01]
       [ 2.83525028e+01 -1.18333216e+01 -3.66135216e+00 ... -1.12477741e+01
        -2.04284859e+01 -5.57233467e+01]
       [ 5.29128671e-01  9.81116772e+00  1.87184060e+00 ... -9.78838801e-01
         3.76280475e+00  5.31671762e+00]]
    
      ...
    
      [[ 1.54199874e+00  5.75790739e+00 -4.94103241e+00 ...  7.93236673e-01
         3.94167393e-01 -4.05623388e+00]
       [ 4.87889004e+00  1.46831703e+01 -1.26672287e+01 ...  2.87703457e+01
         4.06937408e+01 -1.64108810e+01]
       [-1.67135918e+00 -1.01209764e+01 -9.44172287e+00 ...  1.04062510e+01
         5.93833351e+00 -2.48916030e+00]
       [ 4.42830706e+00  1.05754261e+01 -1.89143829e+01 ...  1.97335033e+01
        -5.25908241e+01 -5.49303007e+00]
       [ 1.39856091e+01 -8.07870960e+00  3.68543472e+01 ...  3.88683205e+01
        -1.52968988e+01  3.96382408e+01]]
    
      [[-3.75301689e-01 -7.61647177e+00 -1.49281521e+01 ... -2.09514523e+01
        -5.49276412e-01  1.98785610e+01]
       [-1.03244963e+01 -1.80076046e+01  3.32788315e+01 ... -1.50613260e+01
        -3.02193031e+01 -2.02070961e+01]
       [-2.12092662e+00  7.17105627e+00 -9.41855371e-01 ...  6.25867248e-02
         5.31926441e+00  5.75034046e+00]
       [-3.11814575e+01  5.28341103e+00 -2.61369199e-01 ...  1.68219738e+01
         2.85586429e+00  1.87262535e+01]
       [ 1.80238223e+00 -5.47377110e-01 -1.60285053e+01 ... -1.55820742e+01
         3.95231934e+01 -1.86399841e+01]]
    
      [[ 1.68105745e+00  1.43186867e+00  5.14188623e+00 ... -1.36895399e+01
         4.87276793e+00 -9.86269283e+00]
       [-1.12403259e+01  2.92864819e+01  1.58815947e+01 ...  5.05352936e+01
         2.07584457e+01 -3.45590711e+00]
       [ 1.21873155e-01 -2.34270239e+00  2.35914993e+00 ... -1.10175455e+00
         2.50036597e-01  4.91599664e-02]
       [-4.61994350e-01  1.08560252e+00  8.74766484e-02 ... -4.12061214e+00
         1.23289871e+00  3.91162205e+00]
       [-3.23663673e+01  1.95039883e+01 -1.15742092e+01 ... -1.23621578e+01
        -2.07921581e+01  1.00636253e+01]]]
    
    
     [[[ 5.96356392e+00  3.53006673e+00 -2.87073404e-01 ... -3.55749893e+00
         4.82700348e+00  2.19519567e+00]
       [ 8.93316984e-01 -1.29544973e-01  7.14868069e-01 ... -7.41839349e-01
        -5.54329827e-02 -2.74662703e-01]
       [-8.97530317e-02 -2.22746873e+00  3.71186209e+00 ...  3.03062439e+00
        -1.74851215e+00 -7.95535183e+00]
       [-2.04355354e+01  3.07338309e+00 -8.05264378e+00 ...  2.00888958e+01
        -4.20539856e+00 -4.08923149e+00]
       [ 4.03781557e+00 -1.27447386e+01 -1.88487263e+01 ...  8.59393883e+00
         1.80555420e+01 -5.04499435e+00]]
    
      [[ 1.12149388e-01 -1.69789279e+00  3.18161893e+00 ...  2.84313536e+00
        -5.67727923e-01 -4.69106483e+00]
       [-2.40868683e+01  1.24444580e+00  1.02969179e+01 ... -1.77523022e+01
         1.34223461e+01  4.48354378e+01]
       [ 3.35814781e+01  3.03362346e+00  9.70490837e+00 ... -1.94511452e+01
        -5.02878036e+01 -7.72130156e+00]
       [ 7.33199215e+00  1.88397229e+00  9.72995441e-03 ...  1.80122331e-01
        -4.87740219e-01 -6.78020656e-01]
       [ 2.95233107e+00 -5.20664072e+00  1.14868768e-03 ... -3.03939557e+00
         8.33076286e+00  6.02193880e+00]]
    
      [[-9.76412201e+00 -1.52539234e+01  1.68096581e+01 ...  1.51942329e+01
        -8.12160873e+00 -7.41963434e+00]
       [-2.80317020e+00 -1.28329058e+01  4.18876915e+01 ...  4.03470573e+01
         3.25068626e+01  2.98813381e+01]
       [-3.76550317e-01 -2.16729259e+00  3.18925381e+01 ...  2.47905369e+01
         6.92581594e-01 -9.82490242e-01]
       [-9.92222404e+00  5.50430870e+00  1.68600521e+01 ...  1.35424614e+00
         1.97744325e-01 -2.19476223e+00]
       [-2.38397646e+00 -4.57132263e+01  5.89701157e+01 ...  4.68731356e+00
         3.20474701e+01 -1.96924534e+01]]
    
      ...
    
      [[ 8.98615897e-01  4.12035018e-01  2.38310194e+00 ... -8.04220676e-01
        -7.59798884e-01 -5.71229637e-01]
       [ 1.47286348e+01 -4.35162621e+01  5.27168999e+01 ...  1.14263935e+01
         2.47341957e+01  4.07327385e+01]
       [ 4.66555672e+01  4.92483101e+01  3.63678474e+01 ... -2.89375000e+01
        -2.14916801e+01  6.24550247e+01]
       [ 1.65121574e+01  7.93635845e+00 -1.59465313e+01 ...  2.83100510e+01
         3.14272881e+01 -4.37921953e+00]
       [-8.96559906e+00 -5.18666887e+00  1.05031419e+00 ... -1.48437560e+00
         7.14243793e+00 -1.58808792e+00]]
    
      [[ 5.43379211e+00 -1.33641863e+01  2.35262985e+01 ...  2.39227066e+01
        -2.56763363e+01  2.86671185e+00]
       [ 4.84935837e+01 -3.30319519e+01  1.66667414e+00 ...  3.75582924e+01
         2.59576855e+01  3.51054153e+01]
       [ 4.06705379e+00 -1.48189449e+01  7.20309782e+00 ...  9.09091282e+00
        -4.88039434e-01 -3.68395901e+00]
       [ 7.97676802e-01  5.93055189e-01  1.07620239e+00 ...  2.01554924e-01
         8.56950760e-01 -7.13284872e-03]
       [-4.24230289e+00 -5.53474569e+00  6.41515970e-01 ... -1.35797608e+00
        -3.87446523e+00 -6.41199112e-01]]
    
      [[ 1.02539539e-01  3.57800078e+00 -2.67677450e+00 ... -4.26709592e-01
         1.70550406e+00 -1.87304676e+00]
       [ 1.87429276e+01 -5.20493746e+00 -1.12223167e+01 ...  1.40294533e+01
         1.55372448e+01  6.52171564e+00]
       [ 9.89645243e-01 -1.94473374e+00 -4.01753902e+00 ... -1.86853027e+00
         6.41616523e-01 -2.48612142e+00]
       [-2.61853237e+01  3.44694862e+01  3.95027199e+01 ... -6.27173376e+00
        -1.56761265e+01 -2.63749962e+01]
       [-5.98192334e-01  1.28751874e+00 -1.18300766e-01 ... -1.50607681e+00
        -1.07386851e+00  7.88177371e-01]]]
    
    
     ...
    
    
     [[[ 2.44648342e+01 -1.20890608e+01 -7.97481346e+00 ... -1.40952253e+01
         2.28522282e+01 -3.51251488e+01]
       [ 5.54651604e+01  1.40526986e+00 -3.08322277e+01 ...  2.25473380e+00
        -2.82201328e+01 -2.03017998e+01]
       [-9.17405546e-01 -1.31913614e+01 -1.47484941e+01 ... -3.12332201e+00
        -1.09976635e+01  5.16792870e+00]
       [-5.90595865e+00 -7.47099543e+00  6.51534500e+01 ...  1.26321955e+01
        -8.38975143e+00 -3.14789605e+00]
       [ 7.30676985e+00  1.13032160e+01  1.50884187e+00 ... -2.74689126e+00
        -6.81398869e+00  2.25618382e+01]]
    
      [[-5.56603928e+01  2.61942863e+01 -6.70796432e+01 ... -2.23652897e+01
        -7.57940826e+01 -6.96867447e+01]
       [-1.35654154e+01  7.10292387e+00  1.93026483e+00 ...  1.55202932e+01
        -1.77963867e+01 -1.23998117e+01]
       [-4.19274559e+01  2.38328552e+01 -1.22217255e+01 ...  4.73917885e+01
        -4.24987888e+00  2.60888309e+01]
       [-1.53796649e+00 -1.45116723e+00  1.26469576e+00 ...  1.30533826e+00
        -1.38272333e+00 -2.03623390e+00]
       [ 2.17387409e+01 -3.09905777e+01  1.11452875e+01 ...  1.39459190e+01
         4.62223358e+01  2.56407681e+01]]
    
      [[-7.07630590e-02 -1.40601667e-02  5.96187040e-02 ...  9.31813493e-02
        -2.20389739e-01  1.05878107e-01]
       [-2.14829693e+01 -6.74994421e+00  1.92372417e+01 ...  2.14439964e+01
        -2.82931385e+01  1.96704903e+01]
       [ 1.77732620e+01  5.86688614e+01  3.21126022e+01 ... -1.10967836e+01
        -1.79378915e+00  5.58129997e+01]
       [ 3.88875508e+00  2.07115383e+01 -8.95348835e+00 ... -2.24276867e+01
         1.30070620e+01  4.29913998e+00]
       [-1.14712820e+01  2.73411536e+00  7.80592442e-01 ... -1.57601995e+01
         3.56337523e+00  4.33378696e-01]]
    
      ...
    
      [[ 1.18200541e+00  1.18786335e+01  2.02919888e+00 ...  3.43209958e+00
        -4.44854069e+00  5.20217061e-01]
       [ 8.14926815e+00  1.72780743e+01  1.79248905e+01 ...  1.55819540e+01
        -5.71451664e+00 -4.25845108e+01]
       [-3.78974266e+01  1.25418158e+01  6.83696938e+00 ...  2.63981533e+01
         6.98476601e+00 -7.08583689e+00]
       [ 8.44205856e+00 -8.24601078e+00  1.56746845e+01 ... -6.58814955e+00
        -1.37777920e+01 -2.54229450e+00]
       [ 9.94796038e-01  3.43888879e-01 -1.52902424e+00 ... -4.57940280e-01
         2.83861756e-01 -2.83467650e-01]]
    
      [[-6.32714033e+00  3.57377005e+00 -4.68300581e-01 ...  1.64015222e+00
        -7.94082499e+00  2.08289766e+00]
       [-6.89883089e+00 -8.42857933e+00 -1.18621778e+01 ... -1.44279499e+01
         2.85058226e-02 -3.25672412e+00]
       [-1.90148602e+01 -3.63679838e+00  1.83264503e+01 ... -5.20166779e+00
        -7.26731777e+00  4.10758018e+01]
       [-8.29415035e+00  3.24624228e+00 -8.45081902e+00 ...  2.85457778e+00
         1.38023150e+00 -1.67485294e+01]
       [ 9.65213966e+00 -4.48848295e+00  1.88920784e+01 ... -2.22584362e+01
        -9.07328415e+00  4.64797821e+01]]
    
      [[ 1.23432798e+01 -1.57380047e+01  2.92844353e+01 ...  1.68508568e+01
        -4.06507912e+01 -1.71706066e+01]
       [ 6.93331671e+00 -4.40396500e+00 -1.03334122e+01 ...  6.06902695e+00
        -1.15557384e+00 -1.12741976e+01]
       [ 4.64171219e+00  3.04274988e+00 -7.20796442e+00 ...  2.05625987e+00
        -4.37238693e+00 -2.47608855e-01]
       [-4.08376515e-01  7.45981634e-01  1.32356554e-01 ...  4.85149294e-01
        -7.59086514e+00 -6.05629539e+00]
       [ 5.93417549e+00  5.97403526e+00 -1.75937653e+01 ...  2.52441235e+01
        -1.44886065e+01  5.46756363e+01]]]
    
    
     [[[ 1.86802788e+01  1.68001366e+01  3.51645660e+00 ...  6.25906258e+01
         9.01220989e+00 -2.11268663e+00]
       [-5.69129562e+00  1.32128229e+01  5.61387825e+00 ...  8.60750961e+00
         1.65295935e+00  8.70075345e-01]
       [-1.23380985e+01  7.66581726e+00 -8.30088139e+00 ...  5.12958622e+00
        -6.28846025e+00  3.20354748e+00]
       [-1.04008141e+01 -1.55890026e+01  2.59220161e+01 ... -4.50394964e+00
        -5.47875500e+00 -1.04067364e+01]
       [-1.93264065e+01 -1.36029844e+01 -4.37411499e+00 ... -1.14975080e+01
         4.46687460e+00 -1.15981941e+01]]
    
      [[-3.67892241e+00  3.65758729e+00  1.32610822e+00 ... -1.84953928e+00
         6.49030256e+00  6.02192831e+00]
       [ 2.37886543e+01  2.03963490e+01 -3.27756958e+01 ...  1.09804087e+01
         1.33124723e+01  3.80825653e+01]
       [-2.29888010e+00 -8.72813988e+00  5.20070982e+00 ... -6.62801921e-01
         3.11442995e+00 -7.23264980e+00]
       [ 4.10712385e+00 -1.77132950e+01 -4.05222511e+00 ... -6.88878727e+00
        -1.99754257e+01  1.87472095e+01]
       [-2.51027131e+00  2.35182333e+00  3.85729980e+00 ...  3.65192389e+00
        -3.19690847e+00 -3.42228174e+00]]
    
      [[ 1.72169094e+01 -2.57964516e+01  1.87601528e+01 ... -1.65851383e+01
        -1.94415150e+01 -1.93896046e+01]
       [ 1.85090438e-01 -7.67176971e-02  1.39804363e-01 ...  7.54895285e-02
        -4.30005835e-03 -6.80020750e-02]
       [-1.88544333e-01  8.19620991e+00  4.68417625e+01 ... -3.32375669e+00
         1.46821880e+01  2.02422199e+01]
       [-1.26347227e+01 -2.70303745e+01 -6.91952896e+00 ...  3.88036842e+01
        -1.09909277e+01  6.84207201e+00]
       [ 3.64561176e+00  7.01857662e+00 -5.96842051e+00 ... -1.00900383e+01
        -1.97072101e+00 -2.30578399e+00]]
    
      ...
    
      [[ 1.66111934e+00  1.30175471e-01 -7.72415519e-01 ... -1.18111515e+00
        -4.39077467e-02 -2.11295462e+00]
       [-1.83943784e+00 -1.60321522e+01  9.44291413e-01 ... -4.78417444e+00
         1.05978699e+01 -8.90891171e+00]
       [ 2.59177036e+01 -9.89792538e+00 -1.63445950e+01 ... -2.58745313e+00
        -1.23663540e+01  4.03598166e+00]
       [-8.62205982e+00 -1.35953293e+01  2.06202869e+01 ... -1.22127962e+01
        -1.43807850e+01  1.68333817e+01]
       [ 6.95047855e+00 -8.42428970e+00 -4.71246433e+00 ...  1.18208134e+00
        -3.01237822e+00 -2.10297203e+01]]
    
      [[ 1.56449056e+00  7.97654033e-01  2.45540371e+01 ... -1.04967012e+01
         6.28038216e+00  8.04115582e+00]
       [ 6.49070692e+00 -8.96765518e+00 -1.06929312e+01 ...  7.68582106e-01
         5.34228504e-01  9.19250965e+00]
       [-1.49157152e+01  2.56456490e+01  7.99784851e+00 ... -9.25544930e+00
         9.84661102e+00  1.42120638e+01]
       [-2.38850975e+01 -2.81107306e+00 -9.21412754e+00 ...  1.87675416e+00
        -1.51972694e+01  2.05032539e+00]
       [-9.51219976e-01  2.02247066e+01 -2.39315152e-01 ... -4.09496384e+01
        -2.99745598e+01  1.20476551e+01]]
    
      [[-2.80306969e+01 -4.53742065e+01 -6.89570999e+00 ... -4.12259865e+01
         1.64931831e+01 -1.92483273e+01]
       [-5.24487877e+01 -2.32022667e+01  3.11564121e+01 ...  1.74478378e+01
        -1.41144544e-01  4.78691444e+01]
       [ 2.58291054e+01  2.87931690e+01  5.90821695e+00 ... -3.99922295e+01
         1.86853161e+01 -2.17538667e+00]
       [ 3.87426686e+00 -6.62612839e+01 -2.68486481e+01 ... -3.33529282e+01
        -1.27680349e+01  2.81087437e+01]
       [-1.17350931e+01 -3.57723236e+00 -6.52598572e+00 ... -1.26599159e+01
        -1.72096329e+01 -1.38809433e+01]]]
    
    
     [[[ 3.25236168e+01 -1.58980999e+01  2.18153071e+00 ...  1.96888924e+01
        -1.33255735e-01  4.66681385e+00]
       [ 1.17258308e+02  2.76045265e+01 -6.01291180e+00 ... -5.50663757e+00
         4.55999374e+01 -1.30387182e+01]
       [ 7.15093613e-01  5.56225702e-02  1.60262331e-01 ... -2.03052974e+00
        -1.64390385e+00  1.17405713e-01]
       [-2.43707981e+01 -1.62581272e+01 -1.78934898e+01 ... -4.74811029e+00
         3.96433520e+00 -2.84191360e+01]
       [ 8.16988850e+00 -1.11471355e+00 -6.93172121e+00 ...  1.22799671e+00
        -2.09993382e+01  8.59565926e+00]]
    
      [[ 6.66247070e-01  2.09822598e+01 -3.62289214e+00 ...  2.80020752e+01
         8.23572636e+00  7.73328352e+00]
       [-1.90315366e+00 -1.11063995e+01  1.56001253e+01 ...  1.73766136e+01
         2.25769444e+01 -4.21618652e+01]
       [ 6.36930990e+00 -4.91308594e+00  3.39502525e+00 ... -2.07229185e+00
        -1.96757793e+01 -7.99079418e+00]
       [-3.66930466e+01 -2.98175430e+01 -2.68366737e+01 ...  2.44007378e+01
         1.79023838e+01  7.79001594e-01]
       [ 7.13700151e+00 -1.42282209e+01  3.21812592e+01 ... -2.13095417e+01
         6.33587112e+01 -2.93384533e+01]]
    
      [[-1.84525986e+01  8.78080940e+00 -3.19044247e+01 ... -3.50620103e+00
        -3.29060593e+01  9.68404007e+00]
       [-1.86190434e+01  1.55756140e+01  4.52414799e+00 ... -9.80166340e+00
         6.98554182e+00 -1.99274874e+00]
       [ 1.22505035e+01 -5.76491928e+00  5.01043367e+00 ...  5.50724268e+00
        -4.25583172e+00 -1.43899364e+01]
       [ 1.25311842e+01 -7.42872095e+00 -1.61855869e+01 ... -2.70396595e+01
        -1.03394947e+01  1.21448631e+01]
       [ 6.05832219e-01 -6.48569250e+00  2.65482807e+00 ... -1.23846376e+00
        -4.78013515e+00  3.96111548e-01]]
    
      ...
    
      [[ 1.17935419e+01 -7.80320501e+00  3.39611359e+01 ... -3.24669337e+00
         3.92579579e+00  1.48993692e+01]
       [ 1.57071078e+00 -1.15548480e+00  1.97038248e-01 ... -1.58108699e+00
        -3.11085135e-01  1.96799409e+00]
       [-2.93559432e+00 -5.24500465e+00 -1.45047873e-01 ...  4.15070915e+00
         3.55951905e-01 -1.98227420e-01]
       [ 8.80506516e+00 -6.51160955e-01 -4.52907085e+00 ...  6.54948902e+00
        -8.14756107e+00  4.09238472e+01]
       [ 1.01976681e+01 -2.15937576e+01  3.74276578e-01 ... -1.11078835e+01
        -1.09632292e+01 -3.18265033e+00]]
    
      [[ 5.21317673e+00 -2.22005486e+00 -4.16423798e+00 ...  9.02456284e+00
         3.65998936e+00  3.08844185e+00]
       [-2.32171078e+01 -6.32288933e-01  9.11160529e-01 ... -3.16172962e+01
        -1.41868515e+01 -2.64369774e+01]
       [ 1.04262614e+00  2.51713085e+00  5.10584927e+00 ...  1.21188521e+00
         2.35915232e+00 -1.83088970e+00]
       [ 1.33733807e+01 -3.59415531e+00  7.30300128e-01 ... -2.27976060e+00
         2.99932432e+00  3.05185032e+00]
       [-3.09196520e+00 -3.41616988e+00  6.49591923e+00 ... -7.86339283e+00
        -6.24229622e+00 -2.98605728e+00]]
    
      [[ 3.30554104e+00  8.66824532e+00 -2.24037399e+01 ... -1.18946433e+00
         2.15270329e+01 -1.25522757e+01]
       [ 2.53643894e+01  2.61351051e+01  3.73635826e+01 ...  2.07485542e+01
        -4.43868065e+01 -2.83804417e+01]
       [-2.17776346e+00  9.81323814e+00  2.76007509e+00 ... -6.64397573e+00
        -2.19730606e+01 -8.92380428e+00]
       [-7.73434401e+00  3.66484451e+01  2.31098831e-01 ... -1.59480591e+01
        -2.37636585e+01 -2.28596306e+01]
       [-1.03925920e+00 -9.32735741e-01 -1.15307367e+00 ... -2.32866359e+00
         5.60451508e-01  2.41734815e+00]]]], shape=(19, 19, 5, 80), dtype=float32)
    ----------
    tf.Tensor(
    [[[20 74  8 67 56]
      [54 35  5 34 67]
      [15  9  3 51 35]
      ...
      [ 2 68 59 11 48]
      [18 37 70 21 77]
      [20 17 48 61  1]]
    
     [[61 57 10 32  9]
      [34 78 45 78 47]
      [27 26 12 25 12]
      ...
      [46  7 17 42 10]
      [56 25 70 74 18]
      [17 64 39  9 60]]
    
     [[12 41 48 24 57]
      [16 26 12 61 68]
      [24 49 39 75 44]
      ...
      [11  2 52 78 14]
      [53 17 39 19 19]
      [20  8 26 56 20]]
    
     ...
    
     [[56  0 14  2 23]
      [11 53 22 12  4]
      [22 47  6 30 46]
      ...
      [30 71  7 12  0]
      [44 27 18  6  8]
      [70 54 64 32 79]]
    
     [[18 20 11 24  5]
      [51 29 68 16 69]
      [72 10 16 11  6]
      ...
      [27  8 60  6 75]
      [ 6 67 17 26 73]
      [47 62 29 62  9]]
    
     [[38  0 14  9 60]
      [ 5 32 41 48 34]
      [16 22 76 76 61]
      ...
      [30 44 67 60 20]
      [77 11  2 66 19]
      [58 53 47 73 75]]], shape=(19, 19, 5), dtype=int64)
    ----------
    tf.Tensor(
    [[[  7.8824973   15.941693     9.270486    10.659197    11.924357  ]
      [ 90.77388      5.974023     1.8933523   29.829315    31.208069  ]
      [  6.537015    41.7416      23.049517    37.349552    13.52168   ]
      ...
      [ 62.409065    33.47416     11.878175    59.977737    21.68427   ]
      [ 10.253807    50.624313    71.08136     21.55704     20.643707  ]
      [ 44.418617     6.1362724   39.162514    53.065334    17.46208   ]]
    
     [[ 22.367271    21.832783    77.65796      4.8592863   47.465206  ]
      [ 14.389843    70.55192     21.278732    21.277258    29.335201  ]
      [ 59.208687    37.98005      6.7214456   74.329056    18.578522  ]
      ...
      [ 26.358503    56.81183     14.4094925   52.354233    66.22701   ]
      [ 77.31863     53.607677    33.536755    27.10302     80.672806  ]
      [ 19.82476     56.045506     2.6147327    8.269213    44.02573   ]]
    
     [[  7.2197466    1.7810565    9.357377    46.37595     26.413982  ]
      [  9.393324    81.667854    60.130894    18.69678     24.221481  ]
      [ 25.780058    68.76154     59.284893    17.85466     73.42264   ]
      ...
      [  3.4685545   52.7169     108.622536    31.427288     9.944213  ]
      [ 50.42938    123.73876     25.1464       2.6283898    8.239826  ]
      [ 14.702635    34.794693     3.8779101   54.102886     3.0838993 ]]
    
     ...
    
     [[ 37.04126     55.46516     16.495195    65.15345     37.856792  ]
      [ 75.21127     24.91322    130.03236      3.204923    54.808575  ]
      [  0.21737593  77.92454     74.90996     52.8758      17.541574  ]
      ...
      [ 15.864363    63.17478     40.139515    27.86825      0.99479604]
      [  7.8512235   17.867058    45.14007     17.843685    51.625187  ]
      [ 43.003468    30.131487    13.95415      9.03589     54.675636  ]]
    
     [[ 82.03234     39.557964    38.441784    31.12349     26.958225  ]
      [ 10.974109   121.73265     16.0631      24.478731     7.875782  ]
      [ 46.384388     0.31460574  51.30812     67.466484    10.386171  ]
      ...
      [  2.9835274   18.169016    38.99532     22.453457    31.928385  ]
      [ 33.383083    20.349884    30.179909    64.61757     48.596184  ]
      [ 66.43874     82.38462     41.78406     58.30063     22.135483  ]]
    
     [[ 62.999557   117.25831      2.7462363   38.776752    31.106777  ]
      [ 57.53178     54.09909     20.21145     41.4501      77.36677   ]
      [ 38.235027    16.453613    17.459026    21.03174     24.043415  ]
      ...
      [ 47.036816     5.0479155   14.405617    46.24312     20.980524  ]
      [  9.024563    48.128166     5.1058493   21.123985    11.307784  ]
      [ 32.89026     86.08291     21.845644    55.814663     7.225125  ]]], shape=(19, 19, 5), dtype=float32)
    ------------------------------
    ******************************
    (1789,)
    (1789, 4)
    (1789,)
    ------------------------------
    tf.Tensor([ 7.8824973 15.941693   9.270486  ... 21.845644  55.814663   7.225125 ], shape=(1789,), dtype=float32)
    ----------
    tf.Tensor(
    [[ 1.9506788   2.40341    -1.5526743  -4.4307437 ]
     [ 1.5622553  -5.4505625   0.8918079   8.20554   ]
     [ 4.6399336   3.2303846   4.431282   -2.202031  ]
     ...
     [-6.772757   -3.4325662  -1.1350865   9.632269  ]
     [-0.3067969   2.830866   -6.0716434  -0.22594893]
     [ 4.322794    1.358497   -0.09959805 -4.498812  ]], shape=(1789, 4), dtype=float32)
    ----------
    tf.Tensor([20 74  8 ... 47 73 75], shape=(1789,), dtype=int64)
    ------------------------------
    scores[2] = 9.270486
    boxes[2] = [ 4.6399336  3.2303846  4.431282  -2.202031 ]
    classes[2] = 8
    scores.shape = (1789,)
    boxes.shape = (1789, 4)
    classes.shape = (1789,)
    [92m All tests passed!


**Expected Output**:

<table>
    <tr>
        <td>
            <b>scores[2]</b>
        </td>
        <td>
           9.270486
        </td>
    </tr>
    <tr>
        <td>
            <b>boxes[2]</b>
        </td>
        <td>
           [ 4.6399336  3.2303846  4.431282  -2.202031 ]
        </td>
    </tr>
    <tr>
        <td>
            <b>classes[2]</b>
        </td>
        <td>
           8
        </td>
    </tr>
        <tr>
        <td>
            <b>scores.shape</b>
        </td>
        <td>
           (1789,)
        </td>
    </tr>
    <tr>
        <td>
            <b>boxes.shape</b>
        </td>
        <td>
           (1789, 4)
        </td>
    </tr>
    <tr>
        <td>
            <b>classes.shape</b>
        </td>
        <td>
           (1789,)
        </td>
    </tr>

</table>

**Note** In the test for `yolo_filter_boxes`, you're using random numbers to test the function.  In real data, the `box_class_probs` would contain non-zero values between 0 and 1 for the probabilities.  The box coordinates in `boxes` would also be chosen so that lengths and heights are non-negative.

<a name='2-3'></a>
### 2.3 - Non-max Suppression

Even after filtering by thresholding over the class scores, you still end up with a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS). 

<img src="nb_images/non-max-suppression.png" style="width:500px;height:400;">
<caption><center> <u> <b>Figure 7</b> </u>: In this example, the model has predicted 3 cars, but it's actually 3 predictions of the same car. Running non-max suppression (NMS) will select only the most accurate (highest probability) of the 3 boxes. <br> </center></caption>


Non-max suppression uses the very important function called **"Intersection over Union"**, or IoU.
<img src="nb_images/iou.png" style="width:500px;height:400;">
<caption><center> <u> <b>Figure 8</b> </u>: Definition of "Intersection over Union". <br> </center></caption>

<a name='ex-2'></a>
### Exercise 2 - iou

Implement `iou()` 

Some hints:
- This code uses the convention that (0,0) is the top-left corner of an image, (1,0) is the upper-right corner, and (1,1) is the lower-right corner. In other words, the (0,0) origin starts at the top left corner of the image. As x increases, you move to the right.  As y increases, you move down.
- For this exercise, a box is defined using its two corners: upper left $(x_1, y_1)$ and lower right $(x_2,y_2)$, instead of using the midpoint, height and width. This makes it a bit easier to calculate the intersection.
- To calculate the area of a rectangle, multiply its height $(y_2 - y_1)$ by its width $(x_2 - x_1)$. Since $(x_1,y_1)$ is the top left and $x_2,y_2$ are the bottom right, these differences should be non-negative.
- To find the **intersection** of the two boxes $(xi_{1}, yi_{1}, xi_{2}, yi_{2})$: 
    - Feel free to draw some examples on paper to clarify this conceptually.
    - The top left corner of the intersection $(xi_{1}, yi_{1})$ is found by comparing the top left corners $(x_1, y_1)$ of the two boxes and finding a vertex that has an x-coordinate that is closer to the right, and y-coordinate that is closer to the bottom.
    - The bottom right corner of the intersection $(xi_{2}, yi_{2})$ is found by comparing the bottom right corners $(x_2,y_2)$ of the two boxes and finding a vertex whose x-coordinate is closer to the left, and the y-coordinate that is closer to the top.
    - The two boxes **may have no intersection**.  You can detect this if the intersection coordinates you calculate end up being the top right and/or bottom left corners of an intersection box.  Another way to think of this is if you calculate the height $(y_2 - y_1)$ or width $(x_2 - x_1)$ and find that at least one of these lengths is negative, then there is no intersection (intersection area is zero).  
    - The two boxes may intersect at the **edges or vertices**, in which case the intersection area is still zero.  This happens when either the height or width (or both) of the calculated intersection is zero.


**Additional Hints**

- `xi1` = **max**imum of the x1 coordinates of the two boxes
- `yi1` = **max**imum of the y1 coordinates of the two boxes
- `xi2` = **min**imum of the x2 coordinates of the two boxes
- `yi2` = **min**imum of the y2 coordinates of the two boxes
- `inter_area` = You can use `max(height, 0)` and `max(width, 0)`



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: iou

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """


    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    ### START CODE HERE
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ##(≈ 7 lines)
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = xi2 - xi1
    inter_height =  yi2 - yi1
    inter_area = inter_width * inter_height
    
    if inter_width < 0 or inter_height < 0:
        inter_area = 0
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ## (≈ 3 lines)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area
    ### END CODE HERE
    
    return iou
```


```python
# BEGIN UNIT TEST
## Test case 1: boxes intersect
box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)

print("iou for intersecting boxes = " + str(iou(box1, box2)))
assert iou(box1, box2) < 1, "The intersection area must be always smaller or equal than the union area."
assert np.isclose(iou(box1, box2), 0.14285714), "Wrong value. Check your implementation. Problem with intersecting boxes"

## Test case 2: boxes do not intersect
box1 = (1,2,3,4)
box2 = (5,6,7,8)
print("iou for non-intersecting boxes = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection must be 0"

## Test case 3: boxes intersect at vertices only
box1 = (1,1,2,2)
box2 = (2,2,3,3)
print("iou for boxes that only touch at vertices = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection at vertices must be 0"

## Test case 4: boxes intersect at edge only
box1 = (1,1,3,3)
box2 = (2,3,3,4)
print("iou for boxes that only touch at edges = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection at edges must be 0"

print("\033[92m All tests passed!")
# END UNIT TEST
```

    iou for intersecting boxes = 0.14285714285714285
    iou for non-intersecting boxes = 0.0
    iou for boxes that only touch at vertices = 0.0
    iou for boxes that only touch at edges = 0.0
    [92m All tests passed!


**Expected Output**:

```
iou for intersecting boxes = 0.14285714285714285
iou for non-intersecting boxes = 0.0
iou for boxes that only touch at vertices = 0.0
iou for boxes that only touch at edges = 0.0
```

<a name='2-4'></a>
### 2.4 - YOLO Non-max Suppression

You are now ready to implement non-max suppression. The key steps are: 
1. Select the box that has the highest score.
2. Compute the overlap of this box with all other boxes, and remove boxes that overlap significantly (iou >= `iou_threshold`).
3. Go back to step 1 and iterate until there are no more boxes with a lower score than the currently selected box.

This will remove all boxes that have a large overlap with the selected boxes. Only the "best" boxes remain.

<a name='ex-3'></a>
### Exercise 3 - yolo_non_max_suppression

Implement `yolo_non_max_suppression()` using TensorFlow. TensorFlow has two built-in functions that are used to implement non-max suppression (so you don't actually need to use your `iou()` implementation):

**Reference documentation**: 

- [tf.image.non_max_suppression()](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression)
```
tf.image.non_max_suppression(
    boxes,
    scores,
    max_output_size,
    iou_threshold=0.5,
    name=None
)
```
Note that in the version of TensorFlow used here, there is no parameter `score_threshold` (it's shown in the documentation for the latest version) so trying to set this value will result in an error message: *got an unexpected keyword argument `score_threshold`.*

- [tf.gather()](https://www.tensorflow.org/api_docs/python/tf/gather)
```
keras.gather(
    reference,
    indices
)
```


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: yolo_non_max_suppression

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()

    ### START CODE HERE
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    ##(≈ 1 line)
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    
    # Use tf.gather() to select only nms_indices from scores, boxes and classes
    ##(≈ 3 lines)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    ### END CODE HERE

    
    return scores, boxes, classes
```


```python
# BEGIN UNIT TEST
tf.random.set_seed(10)
scores = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
boxes = tf.random.normal([54, 4], mean=1, stddev=4, seed = 1)
classes = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

assert type(scores) == EagerTensor, "Use tensoflow functions"
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.numpy().shape))
print("boxes.shape = " + str(boxes.numpy().shape))
print("classes.shape = " + str(classes.numpy().shape))

assert type(scores) == EagerTensor, "Use tensoflow functions"
assert type(boxes) == EagerTensor, "Use tensoflow functions"
assert type(classes) == EagerTensor, "Use tensoflow functions"

assert scores.shape == (10,), "Wrong shape"
assert boxes.shape == (10, 4), "Wrong shape"
assert classes.shape == (10,), "Wrong shape"

assert np.isclose(scores[2].numpy(), 8.147684), "Wrong value on scores"
assert np.allclose(boxes[2].numpy(), [ 6.0797963, 3.743308, 1.3914018, -0.34089637]), "Wrong value on boxes"
assert np.isclose(classes[2].numpy(), 1.7079165), "Wrong value on classes"

print("\033[92m All tests passed!")
# END UNIT TEST
```

    scores[2] = 8.147684
    boxes[2] = [ 6.0797963   3.743308    1.3914018  -0.34089637]
    classes[2] = 1.7079165
    scores.shape = (10,)
    boxes.shape = (10, 4)
    classes.shape = (10,)
    [92m All tests passed!


**Expected Output**:

<table>
    <tr>
        <td>
            <b>scores[2]</b>
        </td>
        <td>
           8.147684
        </td>
    </tr>
    <tr>
        <td>
            <b>boxes[2]</b>
        </td>
        <td>
           [ 6.0797963   3.743308    1.3914018  -0.34089637]
        </td>
    </tr>
    <tr>
        <td>
            <b>classes[2]</b>
        </td>
        <td>
           1.7079165
        </td>
    </tr>
        <tr>
        <td>
            <b>scores.shape</b>
        </td>
        <td>
           (10,)
        </td>
    </tr>
    <tr>
        <td>
            <b>boxes.shape</b>
        </td>
        <td>
           (10, 4)
        </td>
    </tr>
    <tr>
        <td>
            <b>classes.shape</b>
        </td>
        <td>
           (10,)
        </td>
    </tr>

</table>

<a name='2-5'></a>
### 2.5 - Wrapping Up the Filtering

It's time to implement a function taking the output of the deep CNN (the 19x19x5x85 dimensional encoding) and filtering through all the boxes using the functions you've just implemented. 

<a name='ex-4'></a>
### Exercise 4 - yolo_eval

Implement `yolo_eval()` which takes the output of the YOLO encoding and filters the boxes using score threshold and NMS. There's just one last implementational detail you have to know. There're a few ways of representing boxes, such as via their corners or via their midpoint and height/width. YOLO converts between a few such formats at different times, using the following functions (which are provided): 

```python
boxes = yolo_boxes_to_corners(box_xy, box_wh) 
```
which converts the yolo box coordinates (x,y,w,h) to box corners' coordinates (x1, y1, x2, y2) to fit the input of `yolo_filter_boxes`
```python
boxes = scale_boxes(boxes, image_shape)
```
YOLO's network was trained to run on 608x608 images. If you are testing this data on a different size image -- for example, the car detection dataset had 720x1280 images -- this step rescales the boxes so that they can be plotted on top of the original 720x1280 image.  

Don't worry about these two functions; you'll see where they need to be called below.  


```python
def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

```


```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: yolo_eval

def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    ### START CODE HERE
    # Retrieve outputs of the YOLO model (≈1 line)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs[0], yolo_outputs[1], yolo_outputs[2], yolo_outputs[3]
    
    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)
    
    # Use one of the functions you've implemented to perform Non-max suppression with 
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)
    ### END CODE HERE
    
    return scores, boxes, classes
```


```python
# BEGIN UNIT TEST
tf.random.set_seed(10)
yolo_outputs = (tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
scores, boxes, classes = yolo_eval(yolo_outputs)
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.numpy().shape))
print("boxes.shape = " + str(boxes.numpy().shape))
print("classes.shape = " + str(classes.numpy().shape))

assert type(scores) == EagerTensor, "Use tensoflow functions"
assert type(boxes) == EagerTensor, "Use tensoflow functions"
assert type(classes) == EagerTensor, "Use tensoflow functions"

assert scores.shape == (10,), "Wrong shape"
assert boxes.shape == (10, 4), "Wrong shape"
assert classes.shape == (10,), "Wrong shape"
    
assert np.isclose(scores[2].numpy(), 171.60194), "Wrong value on scores"
assert np.allclose(boxes[2].numpy(), [-1240.3483, -3212.5881, -645.78, 2024.3052]), "Wrong value on boxes"
assert np.isclose(classes[2].numpy(), 16), "Wrong value on classes"
    
print("\033[92m All tests passed!")
# END UNIT TEST
```

    ------------------------------
    (19, 19, 5, 80)
    (19, 19, 5)
    (19, 19, 5)
    ------------------------------
    tf.Tensor(
    [[[[ 2.55312881e+01  3.86680222e+01 -3.75363898e+00 ...  3.98435616e+00
        -1.00319624e+01  6.18537855e+00]
       [-5.38038254e+00 -5.19427204e+00 -5.80103016e+00 ... -5.78927612e+00
         3.05314231e+00  9.51188278e+00]
       [ 4.10485954e+01 -3.83835733e-01  5.45448303e-01 ...  5.81579351e+00
        -2.18916721e+01 -1.08827362e+01]
       [-9.85832393e-01 -2.18423882e+01 -8.14033318e+00 ...  1.86205273e+01
         2.24296932e+01  2.42106838e+01]
       [-9.89301014e+00  1.66814375e+00  2.99745536e+00 ...  2.98945713e+00
         2.63497424e+00  2.50506306e+00]]
    
      [[ 3.36711025e+00 -2.43515944e+00  1.62909241e+01 ... -9.98916924e-01
        -7.77176428e+00 -9.02505302e+00]
       [-3.69750917e-01  2.95091105e+00  4.47860289e+00 ... -9.07282829e+00
        -8.40966415e+00 -2.49452806e+00]
       [ 8.05798817e+00 -8.28883057e+01 -1.05943741e+02 ... -3.29052925e+01
        -4.67756767e+01  7.66044464e+01]
       [ 2.36913934e-01 -2.51002431e-01  4.95292759e+00 ... -9.81974316e+00
        -6.21456528e+00 -3.80883276e-01]
       [-1.76954627e+00  1.24694357e+01  7.33521748e+00 ... -4.84947395e+00
         2.18769913e+01  1.43743725e+01]]
    
      [[ 5.79053211e+00  3.26662445e+01 -1.22642765e+01 ... -2.77355909e+00
        -3.71534843e+01  2.48811493e+01]
       [-7.21918881e-01 -9.68151808e-01 -8.14786136e-01 ...  1.01718709e-01
        -1.41262078e+00 -5.96707702e-01]
       [ 9.56472015e+00 -1.49380331e+01  4.66050005e+00 ... -7.82875538e-01
         2.15076008e+01  1.52012193e+00]
       [-3.76711321e+00  3.51932411e+01  1.18453455e+01 ... -1.69740372e+01
        -1.01755562e+01  3.43193817e+01]
       [ 3.40570908e+01 -5.68545532e+01 -6.37507019e+01 ...  7.49069166e+00
        -6.02456360e+01  3.76979332e+01]]
    
      ...
    
      [[-4.08364820e+00  3.46003246e+00  1.27911959e+01 ...  8.50643730e+00
         6.71814203e+00  8.00468445e+00]
       [-9.40636444e+00  6.90674019e+00  6.52317905e+00 ...  2.72625184e+00
         1.16803944e+00  4.08990669e+00]
       [ 1.07462130e+01 -5.10550165e+00 -2.69351578e+01 ... -1.38642673e+01
        -9.35469341e+00 -1.17712851e+01]
       [ 2.17726278e+00  1.83839340e+01 -7.33854818e+00 ...  9.81383264e-01
        -3.11367073e+01  2.07462902e+01]
       [-3.20748787e+01  2.45650711e+01  1.02672226e+02 ...  1.21828566e+01
        -4.01143990e+01  1.64671516e+01]]
    
      [[-1.02853518e+01  1.59459925e+00 -2.60115910e+00 ... -5.78212690e+00
         4.58757448e+00 -1.19496746e+01]
       [ 1.19313908e+01  1.73625507e+01  5.86164904e+00 ... -1.08453579e+01
         1.53596354e+00  1.15453358e+01]
       [-1.01242161e+01  8.07196617e+00  4.29869980e-01 ... -1.02990904e+01
         1.68008578e+00  6.24585199e+00]
       [ 1.26066332e+01  1.80454779e+00 -5.25519371e+00 ...  2.08061314e+00
        -4.19303942e+00 -1.16976957e+01]
       [-7.65741873e+00  3.67827034e+00 -5.15470552e+00 ...  1.64532681e+01
         4.60504442e-01 -2.31937313e+00]]
    
      [[-9.54898930e+00 -1.17586076e+00  1.03912382e+01 ... -3.87420297e+00
        -2.20280394e-01  5.25296783e+00]
       [ 2.06318545e+00 -8.49185809e-02  1.38820127e-01 ...  2.42548923e+01
         8.93560123e+00  5.43706083e+00]
       [ 3.94866991e+00  1.74426281e+00  4.56299007e-01 ... -6.70742941e+00
         5.58237696e+00  3.20574236e+00]
       [ 5.22556000e+01  2.60635471e+01  1.92844601e+01 ... -4.10496674e+01
         6.70986328e+01 -2.81918983e+01]
       [-1.63409729e+01  6.36854858e+01 -6.00026178e+00 ... -1.12948742e+01
        -2.26767120e+01  5.62363472e+01]]]
    
    
     [[[-1.24179573e+01  3.31883049e+00  8.85991192e+00 ...  9.00598586e-01
         6.49838972e+00  4.01702356e+00]
       [-2.16217556e+01  9.04826069e+00 -1.52731152e+01 ...  1.93431416e+01
        -6.30699921e+00 -4.30665016e+01]
       [-3.28389130e+01  3.85765800e+01  3.48726082e+01 ...  6.65977970e-02
         2.49076557e+01 -9.98949623e+00]
       [-7.31734097e-01 -5.50714374e-01 -7.81583607e-01 ...  4.28474456e-01
        -1.97185922e+00  1.06191255e-01]
       [ 2.95826316e-01  9.53964293e-01  2.13733703e-01 ...  1.06477156e-01
        -2.07434773e-01  6.32815778e-01]]
    
      [[-2.42844179e-01 -5.03659401e+01  7.73486185e+00 ... -4.86700678e+00
         1.52581348e+01  4.03021355e+01]
       [-2.02026701e+00  4.12300777e+00 -2.08034587e+00 ...  5.20926619e+00
        -1.32893562e+01  3.49843955e+00]
       [ 2.43334293e+00  4.76939797e-01  1.13508642e+00 ... -3.41499877e+00
        -7.01407075e-01 -3.35760027e-01]
       [-4.36408073e-01  3.65159273e-01  6.76167071e-01 ... -2.08635259e+00
        -2.17402768e+00 -3.39833677e-01]
       [-9.73929024e+00 -5.54761362e+00  1.66433773e+01 ...  9.42351913e+00
         1.59383745e+01  3.73506021e+00]]
    
      [[ 1.20585470e+01 -3.80010567e+01  2.18692074e+01 ...  2.44069080e+01
         4.45139456e+00 -1.85899389e+00]
       [ 2.83930802e+00 -1.43041301e+00  2.82913756e+00 ... -2.64873195e+00
         4.97839689e+00 -2.89343500e+00]
       [-1.78587551e+01 -6.92045403e+00  3.78200111e+01 ...  2.65817108e+01
         2.06186104e+01 -5.88654423e+00]
       [-2.57033749e+01  1.07276707e+01  3.31925225e+00 ...  1.01968346e+01
         1.85197430e+01  5.05168152e+01]
       [ 1.18756962e+00  2.20200615e+01  4.20113564e+00 ... -2.19689345e+00
         8.44519138e+00  1.19327736e+01]]
    
      ...
    
      [[-2.23989987e+00 -8.36390877e+00  7.17732000e+00 ... -1.15225172e+00
        -5.72565615e-01  5.89206553e+00]
       [ 2.94574499e+00  8.86531067e+00 -7.64813805e+00 ...  1.73707752e+01
         2.45698071e+01 -9.90845680e+00]
       [ 2.33345008e+00  1.41302919e+01  1.31819601e+01 ... -1.45285749e+01
        -8.29074097e+00  3.47521448e+00]
       [-4.89565134e+00 -1.16915102e+01  2.09105244e+01 ... -2.18160896e+01
         5.81410294e+01  6.07274055e+00]
       [-8.53211689e+00  4.92853022e+00 -2.24835110e+01 ... -2.37121639e+01
         9.33208752e+00 -2.41818638e+01]]
    
      [[ 1.99949026e-01  4.05781841e+00  7.95325327e+00 ...  1.11622791e+01
         2.92637348e-01 -1.05906773e+01]
       [ 4.35119104e+00  7.58918667e+00 -1.40251436e+01 ...  6.34749651e+00
         1.27357264e+01  8.51614761e+00]
       [-4.66597080e+00  1.57760925e+01 -2.07205153e+00 ...  1.37688786e-01
         1.17022104e+01  1.26505632e+01]
       [ 2.36462345e+01 -4.00663662e+00  1.98207468e-01 ... -1.27568226e+01
        -2.16572428e+00 -1.42009201e+01]
       [-9.04966772e-01  2.74835199e-01  8.04782963e+00 ...  7.82367849e+00
        -1.98443909e+01  9.35903931e+00]]
    
      [[-3.76072407e+00 -3.20325947e+00 -1.15030069e+01 ...  3.06251183e+01
        -1.09009571e+01  2.20640087e+01]
       [ 4.69600439e+00 -1.22353611e+01 -6.63504219e+00 ... -2.11127300e+01
        -8.67250252e+00  1.44381535e+00]
       [-2.60833430e+00  5.01386108e+01 -5.04906235e+01 ...  2.35797977e+01
        -5.35129356e+00 -1.05212367e+00]
       [-5.23588371e+00  1.23033724e+01  9.91392136e-01 ... -4.66998062e+01
         1.39727125e+01  4.43312759e+01]
       [-2.25470867e+01  1.35868855e+01 -8.06283569e+00 ... -8.61173725e+00
        -1.44842520e+01  7.01053143e+00]]]
    
    
     [[[ 1.59247780e+01  9.42649937e+00 -7.66585290e-01 ... -9.49975204e+00
         1.28897686e+01  5.86193132e+00]
       [-2.94755673e+00  4.27441955e-01 -2.35875320e+00 ...  2.44774652e+00
         1.82904691e-01  9.06267285e-01]
       [ 4.26511168e-02  1.05850506e+00 -1.76389658e+00 ... -1.44016874e+00
         8.30902219e-01  3.78042555e+00]
       [ 5.61934805e+00 -8.45116556e-01  2.21430993e+00 ... -5.52402925e+00
         1.15639734e+00  1.12445378e+00]
       [ 1.66419697e+00 -5.25277996e+00 -7.76855564e+00 ...  3.54201627e+00
         7.44164228e+00 -2.07930875e+00]]
    
      [[ 2.40076870e-01 -3.63465905e+00  6.81085396e+00 ...  6.08626604e+00
        -1.21532834e+00 -1.00421066e+01]
       [ 7.46985626e+00 -3.85929465e-01 -3.19329596e+00 ...  5.50537109e+00
        -4.16255856e+00 -1.39044352e+01]
       [-1.87984524e+01 -1.69818103e+00 -5.43267536e+00 ...  1.08884859e+01
         2.81504269e+01  4.32227945e+00]
       [ 6.45902276e-01  1.65966079e-01  8.57147621e-04 ...  1.58676412e-02
        -4.29668352e-02 -5.97293451e-02]
       [-1.17173576e+00  2.06643724e+00 -4.55896807e-04 ...  1.20629036e+00
        -3.30635428e+00 -2.39001679e+00]]
    
      [[-3.49715257e+00 -5.46339941e+00  6.02060699e+00 ...  5.44202089e+00
        -2.90886426e+00 -2.65744281e+00]
       [-1.88427353e+00 -8.62619972e+00  2.81566448e+01 ...  2.71210403e+01
         2.18509102e+01  2.00860500e+01]
       [-4.01586056e-01 -2.31138968e+00  3.40129814e+01 ...  2.64387875e+01
         7.38629341e-01 -1.04781318e+00]
       [-8.56084919e+00  4.74909258e+00  1.45467768e+01 ...  1.16843736e+00
         1.70612901e-01 -1.89363074e+00]
       [ 7.87997186e-01  1.51100044e+01 -1.94919243e+01 ... -1.54934001e+00
        -1.05929384e+01  6.50912428e+00]]
    
      ...
    
      [[ 7.54105031e-01  3.45773637e-01  1.99986362e+00 ... -6.74890041e-01
        -6.37611866e-01 -4.79367375e-01]
       [ 3.77649903e+00 -1.11577969e+01  1.35168896e+01 ...  2.92978716e+00
         6.34197760e+00  1.04440880e+01]
       [-6.28115368e+00 -6.63020992e+00 -4.89613676e+00 ...  3.89580250e+00
         2.89338565e+00 -8.40820599e+00]
       [ 1.40361204e+01  6.74628210e+00 -1.35553112e+01 ...  2.40648899e+01
         2.67146893e+01 -3.72254491e+00]
       [ 9.28567982e+00  5.37183762e+00 -1.08781147e+00 ...  1.53736925e+00
        -7.39742994e+00  1.64478421e+00]]
    
      [[-3.71293879e+00  9.13181877e+00 -1.60756435e+01 ... -1.63465099e+01
         1.75447769e+01 -1.95883930e+00]
       [ 1.06198606e+01 -7.23383856e+00  3.64993602e-01 ...  8.22508526e+00
         5.68460798e+00  7.68791676e+00]
       [ 5.02125835e+00 -1.82957363e+01  8.89307404e+00 ...  1.12238054e+01
        -6.02542281e-01 -4.54828215e+00]
       [ 2.52834549e+01  1.87976933e+01  3.41117020e+01 ...  6.38855839e+00
         2.71622219e+01 -2.26085380e-01]
       [ 2.22485828e+01  2.90267467e+01 -3.36440420e+00 ...  7.12184954e+00
         2.03194733e+01  3.36274242e+00]]
    
      [[ 6.47678822e-02  2.26000166e+00 -1.69075274e+00 ... -2.69526035e-01
         1.07726133e+00 -1.18308771e+00]
       [ 5.20397797e+01 -1.44515209e+01 -3.11587887e+01 ...  3.89528122e+01
         4.31391945e+01  1.81075592e+01]
       [ 1.70234013e+00 -3.34523749e+00 -6.91077757e+00 ... -3.21415591e+00
         1.10367787e+00 -4.27650642e+00]
       [ 1.14567213e+01 -1.50812464e+01 -1.72834091e+01 ...  2.74403715e+00
         6.85868979e+00  1.15397072e+01]
       [ 2.27764583e+00 -4.90228891e+00  4.50435817e-01 ...  5.73445940e+00
         4.08880568e+00 -3.00102282e+00]]]
    
    
     ...
    
    
     [[[-1.83085651e+01  9.04699993e+00  5.96805191e+00 ...  1.05483389e+01
        -1.71017532e+01  2.62863464e+01]
       [-2.38878231e+01 -6.05223894e-01  1.32788725e+01 ... -9.71072376e-01
         1.21538916e+01  8.74361134e+00]
       [-2.61728287e+00 -3.76338730e+01 -4.20762444e+01 ... -8.91058159e+00
        -3.13754330e+01  1.47436771e+01]
       [ 7.43229580e+00  9.40180111e+00 -8.19917221e+01 ... -1.58968630e+01
         1.05580006e+01  3.96143913e+00]
       [ 5.54132175e+00  8.57215405e+00  1.14427829e+00 ... -2.08319259e+00
        -5.16760588e+00  1.71104889e+01]]
    
      [[-1.31496239e+00  6.18833244e-01 -1.58473921e+00 ... -5.28374195e-01
        -1.79061568e+00 -1.64633131e+00]
       [-2.17904949e+01  1.14096203e+01  3.10063696e+00 ...  2.49306679e+01
        -2.85868206e+01 -1.99181538e+01]
       [ 1.65719643e+01 -9.42001438e+00  4.83067703e+00 ... -1.87317600e+01
         1.67977846e+00 -1.03116961e+01]
       [ 3.64415789e+00  3.43849015e+00 -2.99665260e+00 ... -3.09295344e+00
         3.27631450e+00  4.82478523e+00]
       [ 1.48546638e+01 -2.11766911e+01  7.61587286e+00 ...  9.52962017e+00
         3.15849590e+01  1.75210228e+01]]
    
      [[ 1.13287334e+01  2.25094652e+00 -9.54459000e+00 ... -1.49177637e+01
         3.52830505e+01 -1.69504375e+01]
       [ 2.51011300e+00  7.88676918e-01 -2.24771786e+00 ... -2.50555921e+00
         3.30582666e+00 -2.29833913e+00]
       [ 6.49682426e+00  2.14457684e+01  1.17384148e+01 ... -4.05630922e+00
        -6.55700266e-01  2.04018383e+01]
       [ 6.76557302e+00  3.60334930e+01 -1.55770874e+01 ... -3.90192108e+01
         2.26294079e+01  7.47955227e+00]
       [ 2.53341007e+01 -6.03823996e+00 -1.72392297e+00 ...  3.48060913e+01
        -7.86964417e+00 -9.57108319e-01]]
    
      ...
    
      [[ 1.73753798e+00  1.74614906e+01  2.98290515e+00 ...  5.04515743e+00
        -6.53931713e+00  7.64714718e-01]
       [ 5.24753556e-02  1.11258224e-01  1.15423255e-01 ...  1.00336447e-01
        -3.67973298e-02 -2.74213284e-01]
       [ 3.18373084e+00 -1.05362737e+00 -5.74368000e-01 ... -2.21768665e+00
        -5.86784303e-01  5.95275164e-01]
       [-2.04185414e+00  1.99443650e+00 -3.79118657e+00 ...  1.59345496e+00
         3.33239102e+00  6.14896715e-01]
       [ 3.88737564e+01  1.34381828e+01 -5.97498474e+01 ... -1.78949833e+01
         1.10924978e+01 -1.10770960e+01]]
    
      [[ 3.36820054e+00 -1.90246677e+00  2.49295920e-01 ... -8.73121381e-01
         4.22723198e+00 -1.10881329e+00]
       [ 3.98324471e+01  4.86649055e+01  6.84897995e+01 ...  8.33040466e+01
        -1.64586827e-01  1.88036633e+01]
       [ 1.52553606e+01  2.91775322e+00 -1.47030573e+01 ...  4.17322636e+00
         5.83046865e+00 -3.29545479e+01]
       [ 3.91646326e-01 -1.53286219e-01  3.99044186e-01 ... -1.34791985e-01
        -6.51739612e-02  7.90858567e-01]
       [ 1.02466812e+01 -4.76495934e+00  2.00557709e+01 ... -2.36294861e+01
        -9.63217068e+00  4.93427887e+01]]
    
      [[-1.15380764e+01  1.47113485e+01 -2.73740883e+01 ... -1.57516041e+01
         3.79989700e+01  1.60504951e+01]
       [-7.96060514e+00  5.05648708e+00  1.18644819e+01 ... -6.96825600e+00
         1.32679164e+00  1.29446611e+01]
       [ 1.49167261e+01  9.77825928e+00 -2.31637001e+01 ...  6.60804987e+00
        -1.40512152e+01 -7.95722246e-01]
       [-1.17669213e+00  2.14946437e+00  3.81370932e-01 ...  1.39790452e+00
        -2.18722458e+01 -1.74505520e+01]
       [-1.05718040e+01 -1.06428146e+01  3.13435020e+01 ... -4.49727058e+01
         2.58116245e+01 -9.74052963e+01]]]
    
    
     [[[ 2.40303364e+01  2.16117191e+01  4.52357435e+00 ...  8.05166702e+01
         1.15933189e+01 -2.71776295e+00]
       [ 3.80436730e+00 -8.83215904e+00 -3.75261736e+00 ... -5.75372076e+00
        -1.10492671e+00 -5.81605017e-01]
       [-1.55531044e+01  9.66334152e+00 -1.04638872e+01 ...  6.46623087e+00
        -7.92707872e+00  4.03831339e+00]
       [ 6.60942888e+00  9.90637875e+00 -1.64727230e+01 ...  2.86213517e+00
         3.48159695e+00  6.61319256e+00]
       [ 5.59056997e+00  3.93494940e+00  1.26530492e+00 ...  3.32589602e+00
        -1.29213738e+00  3.35502172e+00]]
    
      [[-6.04705858e+00  6.01199007e+00  2.17972922e+00 ... -3.04009461e+00
         1.06681347e+01  9.89826584e+00]
       [ 1.14723806e+01  9.83639812e+00 -1.58064957e+01 ...  5.29544163e+00
         6.42010927e+00  1.83658009e+01]
       [ 6.11906385e+00  2.32321987e+01 -1.38430328e+01 ...  1.76421845e+00
        -8.28985977e+00  1.92515659e+01]
       [-4.79274607e+00  2.06702614e+01  4.72868299e+00 ...  8.03876686e+00
         2.33100224e+01 -2.18767738e+01]
       [ 6.43723249e+00 -6.03091526e+00 -9.89149475e+00 ... -9.36483765e+00
         8.19801617e+00  8.77595329e+00]]
    
      [[ 5.51899958e+00 -8.26923084e+00  6.01369667e+00 ... -5.31648111e+00
        -6.23211241e+00 -6.21547222e+00]
       [-4.22695827e+00  1.75202191e+00 -3.19274831e+00 ... -1.72397387e+00
         9.82015356e-02  1.55298090e+00]
       [-2.11008042e-01  9.17273045e+00  5.24226265e+01 ... -3.71975875e+00
         1.64314651e+01  2.26539364e+01]
       [-1.23791361e+01 -2.64835796e+01 -6.77955437e+00 ...  3.80187263e+01
        -1.07685938e+01  6.70366430e+00]
       [-2.72134805e+00 -5.23917294e+00  4.45526028e+00 ...  7.53193378e+00
         1.47108853e+00  1.72120368e+00]]
    
      ...
    
      [[-1.88686905e+01 -1.47866595e+00  8.77388477e+00 ...  1.34163132e+01
         4.98749077e-01  2.40010986e+01]
       [-5.26081705e+00 -4.58521652e+01  2.70068598e+00 ... -1.36828032e+01
         3.03100510e+01 -2.54796047e+01]
       [ 8.14020813e-01 -3.10873121e-01 -5.13349533e-01 ... -8.12664852e-02
        -3.88401270e-01  1.26761734e-01]
       [ 2.97608566e+00  4.69271421e+00 -7.11752701e+00 ...  4.21550369e+00
         4.96383095e+00 -5.81039667e+00]
       [ 4.70272446e+00 -5.69991159e+00 -3.18847418e+00 ...  7.99801409e-01
        -2.03818846e+00 -1.42288017e+01]]
    
      [[ 3.24457788e+00  1.65424502e+00  5.09223213e+01 ... -2.17689819e+01
         1.30248089e+01  1.66764545e+01]
       [ 3.36274934e+00 -4.64602327e+00 -5.53986597e+00 ...  3.98192197e-01
         2.76776701e-01  4.76251745e+00]
       [ 2.82903624e+00 -4.86416292e+00 -1.51693726e+00 ...  1.75546408e+00
        -1.86758852e+00 -2.69557595e+00]
       [-1.53076239e+01 -1.80157733e+00 -5.90520525e+00 ...  1.20278537e+00
        -9.73971653e+00  1.31402481e+00]
       [-1.72584617e+00  3.66947021e+01 -4.34201509e-01 ... -7.42969818e+01
        -5.43843498e+01  2.18586655e+01]]
    
      [[ 3.21912706e-01  5.21090627e-01  7.91923478e-02 ...  4.73451257e-01
        -1.89412534e-01  2.21053407e-01]
       [-2.39999962e+01 -1.06171055e+01  1.42568369e+01 ...  7.98394156e+00
        -6.45862147e-02  2.19044018e+01]
       [-3.33011746e+00 -3.71227050e+00 -7.61739731e-01 ...  5.15615320e+00
        -2.40907645e+00  2.80470163e-01]
       [ 1.55004740e+00 -2.65103397e+01 -1.07418203e+01 ... -1.33441038e+01
        -5.10833645e+00  1.12459688e+01]
       [ 1.22635899e+01  3.73833537e+00  6.81988764e+00 ...  1.32300634e+01
         1.79846783e+01  1.45060806e+01]]]
    
    
     [[[ 1.17632141e+01 -5.75006008e+00  7.89020836e-01 ...  7.12112188e+00
        -4.81962301e-02  1.68790364e+00]
       [ 6.09604340e+01  1.43510847e+01 -3.12600183e+00 ... -2.86279941e+00
         2.37065659e+01 -6.77858925e+00]
       [-1.13913250e+01 -8.86058450e-01 -2.55295277e+00 ...  3.23460083e+01
         2.61871204e+01 -1.87025380e+00]
       [ 1.18465757e+00  7.90302932e-01  8.69797528e-01 ...  2.30804309e-01
        -1.92705214e-01  1.38144612e+00]
       [-6.93900633e+00  9.46769893e-01  5.88738251e+00 ... -1.04298580e+00
         1.78355618e+01 -7.30063057e+00]]
    
      [[ 9.61950302e-01  3.02949047e+01 -5.23085594e+00 ...  4.04303551e+01
         1.18910236e+01  1.11655798e+01]
       [-3.48307872e+00 -2.03265076e+01  2.85507507e+01 ...  3.18020134e+01
         4.13194580e+01 -7.71630249e+01]
       [-1.93950386e+01  1.49607239e+01 -1.03381128e+01 ...  6.31028748e+00
         5.99142570e+01  2.43325825e+01]
       [ 9.44987488e+00  7.67916727e+00  6.91147804e+00 ... -6.28413057e+00
        -4.61055374e+00 -2.00622961e-01]
       [-2.82245588e+00  5.62680626e+00 -1.27266588e+01 ...  8.42724228e+00
        -2.50563431e+01  1.16024199e+01]]
    
      [[ 7.01243258e+00 -3.33691931e+00  1.21244516e+01 ...  1.33244109e+00
         1.25050964e+01 -3.68016911e+00]
       [-1.18539982e+01  9.91636944e+00  2.88034344e+00 ... -6.24032593e+00
         4.44741440e+00 -1.26870322e+00]
       [ 2.14045277e+01 -1.00726776e+01  8.75441265e+00 ...  9.62245560e+00
        -7.43594456e+00 -2.51426220e+01]
       [-1.63140163e+01  9.67125511e+00  2.10715866e+01 ...  3.52022133e+01
         1.34607143e+01 -1.58110743e+01]
       [-1.09467864e+00  1.17190018e+01 -4.79701042e+00 ...  2.23778105e+00
         8.63722992e+00 -7.15734243e-01]]
    
      ...
    
      [[ 1.96770072e+00 -1.30193055e+00  5.66626644e+00 ... -5.41696548e-01
         6.55001760e-01  2.48589444e+00]
       [ 1.23249698e+00 -9.06679630e-01  1.54610917e-01 ... -1.24063885e+00
        -2.44100630e-01  1.54423511e+00]
       [-7.64886236e+00 -1.36661663e+01 -3.77930701e-01 ...  1.08149147e+01
         9.27453518e-01 -5.16493142e-01]
       [ 9.40270901e+00 -6.95358515e-01 -4.83648205e+00 ...  6.99403572e+00
        -8.70057678e+00  4.37015533e+01]
       [ 1.02711878e+01 -2.17494392e+01  3.76974940e-01 ... -1.11879654e+01
        -1.10422688e+01 -3.20559549e+00]]
    
      [[ 1.24936037e+01 -5.32045746e+00 -9.97977638e+00 ...  2.16277580e+01
         8.77132320e+00  7.40158510e+00]
       [ 2.91652889e+01  7.94280231e-01 -1.14459825e+00 ...  3.97175903e+01
         1.78214970e+01  3.32100830e+01]
       [ 9.02056789e+00  2.17776527e+01  4.41746635e+01 ...  1.04849596e+01
         2.04108582e+01 -1.58404484e+01]
       [-2.63770008e+01  7.08893585e+00 -1.44040823e+00 ...  4.49648809e+00
        -5.91572046e+00 -6.01932049e+00]
       [ 2.94313335e+00  3.25173235e+00 -6.18323803e+00 ...  7.48488808e+00
         5.94182301e+00  2.84232330e+00]]
    
      [[-2.21448803e+00 -5.80713558e+00  1.50089855e+01 ...  7.96860337e-01
        -1.44216499e+01  8.40917301e+00]
       [-3.26178980e+00 -3.36090183e+00 -4.80485296e+00 ... -2.66820645e+00
         5.70801973e+00  3.64964581e+00]
       [-3.06353420e-01  1.38046169e+00  3.88269186e-01 ... -9.34630811e-01
        -3.09102559e+00 -1.25534213e+00]
       [-2.58067966e+00  1.22283020e+01  7.71095902e-02 ... -5.32130861e+00
        -7.92910051e+00 -7.62745762e+00]
       [-1.26369438e+01 -1.13416643e+01 -1.40208788e+01 ... -2.83155460e+01
         6.81484890e+00  2.93939114e+01]]]], shape=(19, 19, 5, 80), dtype=float32)
    ----------
    tf.Tensor(
    [[[20 65 10 59 56]
      [54 35  5 14 67]
      [15  7  3 51 35]
      ...
      [ 2 68 59 50  2]
      [18 37 70 21 77]
      [36 20 64 78  1]]
    
     [[61 57 10 32  9]
      [34 16 18 74 67]
      [27 26 38 40 12]
      ...
      [18  7 13 50 22]
      [ 9 20 70  0 55]
      [61 45 60  9 60]]
    
     [[12 29 49 22 57]
      [16 36 78 61  7]
      [24 49 39 75 63]
      ...
      [11  2 56 78 44]
      [34 17 39 19 49]
      [20  8 26 64 12]]
    
     ...
    
     [[27  2 14 75 23]
      [11 53 38 36  4]
      [78 22  6 30  3]
      ...
      [30 71 46 70  0]
      [60 75 42 27  8]
      [46 25 64 32 53]]
    
     [[18 50 11 25 38]
      [51 29 41 40 66]
      [72 49 16 11  5]
      ...
      [46  8 60 12 75]
      [ 6 67 50 26 73]
      [53 62 77 62 69]]
    
     [[38  0 11 75 19]
      [ 5 32 56 70 29]
      [54 22 76 53 32]
      ...
      [30 44 67 60 20]
      [77 28  2 67 68]
      [31 29 47 73 75]]], shape=(19, 19, 5), dtype=int64)
    ----------
    tf.Tensor(
    [[[ 46.959114    11.018186    82.90598     35.308758    14.8725605 ]
      [ 30.353521    13.463264   119.75863     19.196886    24.543999  ]
      [ 41.438496     1.6955218   33.886448    48.93687     98.430786  ]
      ...
      [ 12.791196    13.520904    26.590979    49.55878    102.672226  ]
      [  8.733143    31.908936    16.027056    25.844324    16.453268  ]
      [ 21.474789    34.861675    11.965935    67.09863     63.685486  ]]
    
     [[ 15.008878    74.44063     54.749607     1.6782255    1.1060759 ]
      [ 73.39165     14.377197     8.748685     1.8465466   26.062887  ]
      [ 34.415287    14.047679    41.91695     81.63951     41.6974    ]
      ...
      [ 25.198488    34.301483    24.848166    67.38335     26.264841  ]
      [ 45.122032    23.124039    73.779785    23.646235    25.209795  ]
      [ 67.48449     16.5409      56.545532    93.71681     30.669243  ]]
    
     [[ 19.27922      8.913833     6.113744    12.566675    10.886597  ]
      [ 20.108177    18.609264    28.150427     1.6470684    7.6114626 ]
      [  9.233478    46.221085    63.226574    15.40492     38.07524   ]
      ...
      [  2.9107592   13.51689     12.2807045   26.71469     12.476953  ]
      [ 25.588875    27.098192    31.046198    83.310394    39.90474   ]
      [  9.286744    96.607544     6.6705947   25.599144     6.7611265 ]]
    
     ...
    
     [[ 30.368525    13.2788725   47.05944     77.18439     28.709908  ]
      [  1.7768469   40.01878     38.87662      6.2176957   37.452164  ]
      [ 35.28305      6.284855    27.382526    91.99219     40.842136  ]
      ...
      [ 23.32048      0.40679964   3.6906881   11.165233    38.873756  ]
      [  6.287334   137.1806      34.616257     1.1343542   54.805134  ]
      [ 59.09677     25.230352    44.84342     26.035927   105.93857   ]]
    
     [[105.52652     15.752232    48.458767    28.83674      9.751446  ]
      [ 18.038183    58.70712     26.227865    46.76449     20.672735  ]
      [ 14.868836     7.826271    57.421116    66.10171      8.681577  ]
      ...
      [ 35.550907    51.963627     1.2247614   10.85528     21.602886  ]
      [ 69.23277     10.543005     5.0436435   41.412495    88.170494  ]
      [  0.6996084   37.69831      5.156153    23.325378    27.55321   ]]
    
     [[ 22.78582     60.960434    65.0187       2.2253165   26.348547  ]
      [ 83.06635     99.01008    103.90444     13.502345    29.294006  ]
      [ 16.55035     10.475355    30.505047    47.126625    23.229609  ]
      ...
      [  7.847886     3.960971    37.534676    49.38187     21.131784  ]
      [ 21.627758    67.87364     44.174664    36.39209      8.060748  ]
      [ 20.574509     5.9465084    3.0731015   18.623398    87.8544    ]]], shape=(19, 19, 5), dtype=float32)
    ------------------------------
    ******************************
    (1786,)
    (1786, 4)
    (1786,)
    ------------------------------
    tf.Tensor([46.959114  11.018186  82.90598   ...  3.0731015 18.623398  87.8544   ], shape=(1786,), dtype=float32)
    ----------
    tf.Tensor(
    [[ -2.3058128   -0.75167537   5.570725     2.5230653 ]
     [ -4.41899     -0.61870974   1.7488999   -1.5749795 ]
     [ -2.2796044   -0.12143171 -10.342308     2.8530235 ]
     ...
     [  0.7958269   -2.8839426   -4.6315866    1.3559202 ]
     [  1.8101821   -2.5126626   -5.6385837   -0.48695254]
     [  5.3965187    0.40965655   0.9595802    0.51460147]], shape=(1786, 4), dtype=float32)
    ----------
    tf.Tensor([20 65 10 ... 47 73 75], shape=(1786,), dtype=int64)
    ------------------------------
    scores[2] = 171.60194
    boxes[2] = [-1240.3483 -3212.5881  -645.78    2024.3052]
    classes[2] = 16
    scores.shape = (10,)
    boxes.shape = (10, 4)
    classes.shape = (10,)
    [92m All tests passed!


**Expected Output**:

<table>
    <tr>
        <td>
            <b>scores[2]</b>
        </td>
        <td>
           171.60194
        </td>
    </tr>
    <tr>
        <td>
            <b>boxes[2]</b>
        </td>
        <td>
           [-1240.3483 -3212.5881  -645.78    2024.3052]
        </td>
    </tr>
    <tr>
        <td>
            <b>classes[2]</b>
        </td>
        <td>
           16
        </td>
    </tr> 
        <tr>
        <td>
            <b>scores.shape</b>
        </td>
        <td>
           (10,)
        </td>
    </tr>
    <tr>
        <td>
            <b>boxes.shape</b>
        </td>
        <td>
           (10, 4)
        </td>
    </tr>
    <tr>
        <td>
            <b>classes.shape</b>
        </td>
        <td>
           (10,)
        </td>
    </tr>

</table>

<a name='3'></a>
## 3 - Test YOLO Pre-trained Model on Images

In this section, you are going to use a pre-trained model and test it on the car detection dataset.  

<a name='3-1'></a>
### 3.1 - Defining Classes, Anchors and Image Shape

You're trying to detect 80 classes, and are using 5 anchor boxes. The information on the 80 classes and 5 boxes is gathered in two files: "coco_classes.txt" and "yolo_anchors.txt". You'll read class names and anchors from text files. The car detection dataset has 720x1280 images, which are pre-processed into 608x608 images.


```python
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
model_image_size = (608, 608) # Same as yolo_model input layer size
```


```python
!cat model_data/coco_classes.txt | wc -l
```

    80



```python
!cat model_data/yolo_anchors.txt
```

    0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828


<a name='3-2'></a>
### 3.2 - Loading a Pre-trained Model

Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes. You are going to load an existing pre-trained Keras YOLO model stored in "yolo.h5". These weights come from the official YOLO website, and were converted using a function written by Allan Zelener. References are at the end of this notebook. Technically, these are the parameters from the "YOLOv2" model, but are simply referred to as "YOLO" in this notebook.

Run the cell below to load the model from this file.


```python
yolo_model = load_model("model_data/", compile=False)
```

This loads the weights of a trained YOLO model. Here's a summary of the layers your model contains:


```python
yolo_model.summary()
```

    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 608, 608, 3) 0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 608, 608, 32) 864         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 608, 608, 32) 128         conv2d[0][0]                     
    __________________________________________________________________________________________________
    leaky_re_lu (LeakyReLU)         (None, 608, 608, 32) 0           batch_normalization[0][0]        
    __________________________________________________________________________________________________
    max_pooling2d (MaxPooling2D)    (None, 304, 304, 32) 0           leaky_re_lu[0][0]                
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 304, 304, 64) 18432       max_pooling2d[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 304, 304, 64) 256         conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)       (None, 304, 304, 64) 0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, 152, 152, 64) 0           leaky_re_lu_1[0][0]              
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 152, 152, 128 73728       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 152, 152, 128 512         conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)       (None, 152, 152, 128 0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 152, 152, 64) 8192        leaky_re_lu_2[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 152, 152, 64) 256         conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)       (None, 152, 152, 64) 0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 152, 152, 128 73728       leaky_re_lu_3[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 152, 152, 128 512         conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_4 (LeakyReLU)       (None, 152, 152, 128 0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, 76, 76, 128)  0           leaky_re_lu_4[0][0]              
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 76, 76, 256)  294912      max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 76, 76, 256)  1024        conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_5 (LeakyReLU)       (None, 76, 76, 256)  0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 76, 76, 128)  32768       leaky_re_lu_5[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, 76, 76, 128)  512         conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_6 (LeakyReLU)       (None, 76, 76, 128)  0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 76, 76, 256)  294912      leaky_re_lu_6[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, 76, 76, 256)  1024        conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_7 (LeakyReLU)       (None, 76, 76, 256)  0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)  (None, 38, 38, 256)  0           leaky_re_lu_7[0][0]              
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 38, 38, 512)  1179648     max_pooling2d_3[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, 38, 38, 512)  2048        conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_8 (LeakyReLU)       (None, 38, 38, 512)  0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 38, 38, 256)  131072      leaky_re_lu_8[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, 38, 38, 256)  1024        conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_9 (LeakyReLU)       (None, 38, 38, 256)  0           batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 38, 38, 512)  1179648     leaky_re_lu_9[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, 38, 38, 512)  2048        conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_10 (LeakyReLU)      (None, 38, 38, 512)  0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 38, 38, 256)  131072      leaky_re_lu_10[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, 38, 38, 256)  1024        conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_11 (LeakyReLU)      (None, 38, 38, 256)  0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 38, 38, 512)  1179648     leaky_re_lu_11[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, 38, 38, 512)  2048        conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_12 (LeakyReLU)      (None, 38, 38, 512)  0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)  (None, 19, 19, 512)  0           leaky_re_lu_12[0][0]             
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 19, 19, 1024) 4718592     max_pooling2d_4[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, 19, 19, 1024) 4096        conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_13 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 19, 19, 512)  524288      leaky_re_lu_13[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, 19, 19, 512)  2048        conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_14 (LeakyReLU)      (None, 19, 19, 512)  0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 19, 19, 1024) 4718592     leaky_re_lu_14[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 19, 19, 1024) 4096        conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_15 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 19, 19, 512)  524288      leaky_re_lu_15[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, 19, 19, 512)  2048        conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_16 (LeakyReLU)      (None, 19, 19, 512)  0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 19, 19, 1024) 4718592     leaky_re_lu_16[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, 19, 19, 1024) 4096        conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_17 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 19, 19, 1024) 9437184     leaky_re_lu_17[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, 19, 19, 1024) 4096        conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, 38, 38, 64)   32768       leaky_re_lu_12[0][0]             
    __________________________________________________________________________________________________
    leaky_re_lu_18 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, 38, 38, 64)   256         conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 19, 19, 1024) 9437184     leaky_re_lu_18[0][0]             
    __________________________________________________________________________________________________
    leaky_re_lu_20 (LeakyReLU)      (None, 38, 38, 64)   0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, 19, 19, 1024) 4096        conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    space_to_depth_x2 (Lambda)      (None, 19, 19, 256)  0           leaky_re_lu_20[0][0]             
    __________________________________________________________________________________________________
    leaky_re_lu_19 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 19, 19, 1280) 0           space_to_depth_x2[0][0]          
                                                                     leaky_re_lu_19[0][0]             
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, 19, 19, 1024) 11796480    concatenate[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, 19, 19, 1024) 4096        conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_21 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, 19, 19, 425)  435625      leaky_re_lu_21[0][0]             
    ==================================================================================================
    Total params: 50,983,561
    Trainable params: 50,962,889
    Non-trainable params: 20,672
    __________________________________________________________________________________________________


**Note**: On some computers, you may see a warning message from Keras. Don't worry about it if you do -- this is fine!

**Reminder**: This model converts a preprocessed batch of input images (shape: (m, 608, 608, 3)) into a tensor of shape (m, 19, 19, 5, 85) as explained in Figure (2).

<a name='3-3'></a>
### 3.3 - Convert Output of the Model to Usable Bounding Box Tensors

The output of `yolo_model` is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial processing and conversion. You will need to call `yolo_head` to format the encoding of the model you got from `yolo_model` into something decipherable:

```python
yolo_model_outputs = yolo_model(image_data) 
yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
```
The variable `yolo_outputs` will be defined as a set of 4 tensors that you can then use as input by your yolo_eval function. If you are curious about how yolo_head is implemented, you can find the function definition in the file `keras_yolo.py`. The file is also located in your workspace in this path: `yad2k/models/keras_yolo.py`.

<a name='3-4'></a>
### 3.4 - Filtering Boxes

`yolo_outputs` gave you all the predicted boxes of `yolo_model` in the correct format. To perform filtering and select only the best boxes, you will call `yolo_eval`, which you had previously implemented, to do so:

    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)

<a name='3-5'></a>
### 3.5 - Run the YOLO on an Image

Let the fun begin! You will create a graph that can be summarized as follows:

`yolo_model.input` is given to `yolo_model`. The model is used to compute the output `yolo_model.output`
`yolo_model.output` is processed by `yolo_head`. It gives you `yolo_outputs`
`yolo_outputs` goes through a filtering function, `yolo_eval`. It outputs your predictions: `out_scores`, `out_boxes`, `out_classes`.

Now, we have implemented for you the `predict(image_file)` function, which runs the graph to test YOLO on an image to compute `out_scores`, `out_boxes`, `out_classes`.

The code below also uses the following function:

    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
which opens the image file and scales, reshapes and normalizes the image. It returns the outputs:

    image: a python (PIL) representation of your image used for drawing boxes. You won't need to use it.
    image_data: a numpy-array representing the image. This will be the input to the CNN.


```python
def predict(image_file):
    """
    Runs the graph to predict boxes for "image_file". Prints and plots the predictions.
    
    Arguments:
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
    
    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    
    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), "images/" + image_file))
    # Generate colors for drawing bounding boxes.
    colors = get_colors_for_classes(len(class_names))
    # Draw bounding boxes on the image file
    #draw_boxes2(image, out_scores, out_boxes, out_classes, class_names, colors, image_shape)
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=100)
    # Display the results in the notebook
    output_image = Image.open(os.path.join("out", image_file))
    imshow(output_image)

    return out_scores, out_boxes, out_classes
```

Run the following cell on the "test.jpg" image to verify that your function is correct.


```python
out_scores, out_boxes, out_classes = predict("test.jpg")
```

    ------------------------------
    (1, 19, 19, 5, 80)
    (1, 19, 19, 5)
    (1, 19, 19, 5)
    ------------------------------
    tf.Tensor(
    [[[[[1.90493699e-06 1.08790603e-08 2.77811978e-07 ... 2.24838203e-07
         1.51926525e-08 1.63996070e-08]
        [2.08107394e-07 1.28075657e-08 4.12293417e-08 ... 1.54668513e-07
         6.56204913e-09 1.63759299e-08]
        [1.21493109e-08 1.65897340e-09 3.04321035e-09 ... 2.89700308e-09
         5.08612541e-10 2.31525799e-09]
        [2.21361862e-09 1.51108692e-09 3.18903548e-09 ... 1.26109279e-09
         7.44297735e-10 1.57885904e-09]
        [1.20934307e-09 5.79959303e-10 7.28539118e-10 ... 6.66053657e-10
         2.04581060e-10 7.86450682e-10]]
    
       [[6.97650023e-07 1.59868137e-08 2.12245268e-07 ... 4.93647967e-09
         2.25675523e-09 6.89016888e-09]
        [6.10403248e-08 2.84339805e-08 1.58323928e-07 ... 8.37057978e-09
         5.59955682e-09 1.81382909e-08]
        [2.58030664e-09 8.26556101e-10 1.10867704e-09 ... 3.32402855e-10
         1.45770590e-10 5.22740851e-10]
        [9.32824373e-10 2.38276482e-10 2.91502683e-10 ... 5.45722842e-11
         6.31697195e-11 9.51317844e-11]
        [1.59183305e-10 3.91173309e-11 4.02711614e-11 ... 1.28287051e-11
         1.05145866e-11 4.36640619e-11]]
    
       [[2.44575773e-07 6.01401462e-09 4.22627231e-08 ... 5.43828182e-09
         7.20733639e-10 1.78215542e-09]
        [4.27442082e-08 5.14618126e-09 1.74672650e-08 ... 6.90279034e-09
         1.02176401e-09 2.94335822e-09]
        [1.10489014e-10 3.29988085e-11 6.40866665e-11 ... 2.62062125e-11
         5.85483485e-12 1.91747139e-11]
        [2.22124408e-09 6.46151244e-10 1.22218857e-09 ... 2.38619152e-10
         1.53591501e-10 2.60411359e-10]
        [2.72368586e-11 7.67882806e-12 1.58433405e-11 ... 4.05659664e-12
         1.69254389e-12 7.18807956e-12]]
    
       ...
    
       [[1.04842549e-07 2.60161137e-09 1.75828969e-08 ... 3.49930538e-08
         7.31243555e-09 4.40222925e-09]
        [1.27395239e-07 1.21779054e-09 1.02530207e-08 ... 4.13773229e-08
         1.22920421e-08 4.63461269e-09]
        [1.08040554e-09 1.58979063e-11 9.50741361e-11 ... 1.58919822e-10
         5.81142912e-11 1.22283447e-10]
        [1.24129851e-09 4.69014549e-11 2.46689419e-10 ... 2.35705594e-10
         1.22905783e-10 1.00375944e-10]
        [4.51762294e-10 4.17257028e-11 9.62687083e-11 ... 9.60495086e-11
         2.52812753e-11 9.54606533e-11]]
    
       [[4.54659386e-08 2.05250750e-09 3.10841592e-08 ... 1.33503937e-08
         4.42800108e-09 3.25548921e-09]
        [1.31849660e-07 2.98294833e-09 7.81045983e-08 ... 3.08016475e-08
         2.14403411e-08 9.58209068e-09]
        [3.19621551e-09 8.63888266e-11 5.09821962e-10 ... 4.38818565e-10
         2.33025516e-10 5.60258562e-10]
        [4.01654016e-10 2.04076929e-11 9.94153510e-11 ... 5.68567034e-11
         4.40439143e-11 4.31165381e-11]
        [1.11603483e-10 1.88140146e-11 2.85987831e-11 ... 2.22808352e-11
         1.16095154e-11 4.95692723e-11]]
    
       [[3.51536914e-06 1.67362160e-07 1.45841250e-06 ... 5.37346182e-07
         8.67698944e-08 9.35896622e-08]
        [1.68789370e-06 7.17357125e-08 5.52245069e-07 ... 2.98029704e-07
         6.18089189e-08 9.45946539e-08]
        [8.94268837e-09 6.40408671e-10 2.23228680e-09 ... 1.34701461e-09
         5.15815557e-10 1.70963155e-09]
        [8.52810877e-10 9.48780499e-11 5.18762810e-10 ... 1.53226085e-10
         1.21978996e-10 1.32868994e-10]
        [6.68646027e-10 1.40172915e-10 2.36901943e-10 ... 1.65095548e-10
         6.31799474e-11 2.33427278e-10]]]
    
    
      [[[9.32368494e-07 3.05016137e-08 2.51426506e-08 ... 9.11281006e-08
         5.95452931e-09 7.69315545e-09]
        [1.73514380e-07 4.45873871e-09 4.33681269e-09 ... 3.68672879e-08
         6.06922124e-10 2.66029998e-09]
        [1.25697079e-08 1.79250970e-09 1.54492963e-09 ... 4.22300994e-09
         1.71778522e-10 6.82826906e-10]
        [3.70253467e-10 4.14727891e-10 2.85265089e-10 ... 2.85535651e-10
         7.12918266e-11 6.83305218e-11]
        [5.55895885e-10 6.01540040e-10 4.13492546e-10 ... 4.90180452e-10
         6.68486794e-11 1.90555169e-10]]
    
       [[3.86983601e-09 7.18451756e-11 1.54215821e-10 ... 1.41313364e-10
         3.88664205e-11 4.74067556e-11]
        [1.79911851e-07 4.28818714e-09 4.35687966e-08 ... 8.42900061e-09
         2.20454632e-09 1.37171929e-09]
        [9.01581174e-08 4.38330616e-09 1.69433338e-08 ... 6.46093579e-09
         1.23729882e-09 2.20895080e-09]
        [4.18497903e-10 2.57688940e-11 5.93596769e-11 ... 1.72166500e-11
         8.88125684e-12 5.46571078e-12]
        [3.16005451e-11 7.80563548e-12 9.99129859e-12 ... 4.93218357e-12
         1.80905106e-12 5.63631173e-12]]
    
       [[4.27395941e-09 1.02340754e-10 2.72946166e-10 ... 6.75057177e-10
         4.42265564e-11 5.80054546e-11]
        [3.90831545e-08 6.24658769e-10 6.65357680e-09 ... 9.27177535e-09
         2.32226932e-10 2.25470920e-10]
        [5.16192733e-09 1.98685485e-10 1.91666727e-09 ... 7.61246732e-10
         5.11969356e-11 9.68968794e-11]
        [2.37285547e-09 2.19704657e-10 1.92362037e-09 ... 2.80805240e-10
         6.67768618e-11 4.41904673e-11]
        [4.88850141e-11 6.43197865e-12 3.51324039e-11 ... 7.87987297e-12
         1.16095978e-12 3.46508499e-12]]
    
       ...
    
       [[7.11315593e-08 6.60654642e-10 3.92143251e-09 ... 1.80260624e-08
         5.56727686e-09 7.35930827e-09]
        [9.41079463e-07 4.11026191e-09 8.04216604e-09 ... 3.43368100e-07
         3.58903982e-08 2.53399470e-08]
        [1.33989744e-07 1.19208177e-09 5.93852478e-09 ... 2.34508928e-08
         7.18638438e-09 2.00998453e-08]
        [6.68879352e-09 8.34322889e-11 3.32881250e-10 ... 1.10403364e-09
         3.77312986e-10 3.01348529e-10]
        [2.50037724e-09 7.30670802e-11 2.86022761e-10 ... 3.09416104e-10
         7.56922927e-11 3.36887324e-10]]
    
       [[3.73929261e-08 5.31641786e-10 1.16628174e-08 ... 1.13535208e-08
         2.97088731e-09 3.01619307e-09]
        [6.26608880e-07 5.98771122e-09 3.68443231e-08 ... 2.20923084e-07
         3.22238556e-08 1.83610886e-08]
        [4.23980602e-08 1.42035650e-09 4.84441776e-09 ... 1.19018777e-08
         3.91501898e-09 1.18481012e-08]
        [1.68349348e-10 8.82660351e-12 3.43851371e-11 ... 7.13128237e-11
         2.64856661e-11 2.18886506e-11]
        [3.76641440e-10 3.38006914e-11 6.76717016e-11 ... 1.09115397e-10
         2.58981118e-11 1.09620348e-10]]
    
       [[8.60771593e-07 1.34069049e-08 1.57095997e-07 ... 1.47363608e-07
         2.29092070e-08 3.06849799e-08]
        [2.53130270e-06 7.51822142e-08 4.48124382e-07 ... 1.14550232e-06
         1.14365278e-07 2.06514457e-07]
        [1.63342602e-08 7.58058449e-10 3.19125459e-09 ... 4.18930313e-09
         1.06531695e-09 6.15679729e-09]
        [1.13915787e-10 6.50282909e-12 4.35653284e-11 ... 4.19376825e-11
         1.98980485e-11 2.72908987e-11]
        [4.06739809e-10 5.42018513e-11 1.31000530e-10 ... 2.24747429e-10
         4.45956362e-11 2.01719877e-10]]]
    
    
      [[[2.09420904e-07 2.58496549e-08 7.23897564e-09 ... 2.00886277e-07
         7.46128670e-09 1.03568807e-08]
        [1.27520013e-06 5.42251293e-08 6.05479089e-09 ... 2.80775055e-07
         3.11390336e-09 1.32197027e-08]
        [2.06635661e-07 1.75150827e-08 3.82620957e-09 ... 1.33446241e-08
         6.93708035e-10 3.54802232e-09]
        [1.24102340e-10 1.90342964e-10 4.38729018e-11 ... 8.51999929e-11
         2.10456305e-11 2.06471836e-11]
        [1.40277945e-09 1.93774752e-09 6.84942825e-10 ... 7.54060758e-10
         9.93706992e-11 3.05435233e-10]]
    
       [[1.94887551e-09 5.15445811e-11 2.24856370e-10 ... 1.94382746e-10
         3.13355175e-11 1.68335318e-10]
        [2.06808451e-07 1.94017313e-09 9.63505009e-09 ... 7.12923764e-09
         7.88592802e-10 1.89695837e-09]
        [3.96821804e-07 8.86446294e-09 1.94116421e-08 ... 2.03235722e-08
         1.46433643e-09 6.78863854e-09]
        [9.02073277e-11 1.03317745e-11 1.24358111e-11 ... 1.02790051e-11
         2.33009207e-12 2.16867865e-12]
        [2.61161953e-10 5.12940940e-11 3.75434649e-11 ... 2.82702421e-11
         3.25316336e-12 1.77834084e-11]]
    
       [[4.31852110e-09 1.04664055e-10 8.68457084e-10 ... 9.65650226e-10
         6.08338657e-11 3.32217809e-10]
        [7.28028269e-08 6.88118118e-10 1.02698356e-08 ... 8.61489546e-09
         3.91879390e-10 1.53474755e-09]
        [1.98569964e-07 1.87617077e-09 1.11524585e-08 ... 1.09495426e-08
         3.93414773e-10 1.00646380e-09]
        [3.70153486e-09 1.38487458e-10 4.78002804e-10 ... 3.03931741e-10
         3.45021511e-11 2.08768402e-11]
        [9.05629405e-10 5.59581062e-11 1.19744825e-10 ... 9.72875600e-11
         4.35167440e-12 1.20976831e-11]]
    
       ...
    
       [[1.97518460e-07 1.10417664e-09 6.84424961e-09 ... 4.89880669e-08
         5.33871658e-09 9.99602179e-09]
        [2.14754508e-07 1.41552359e-09 2.04224859e-09 ... 7.81618112e-08
         3.99958466e-09 4.76981032e-09]
        [5.04209822e-07 3.28453420e-09 8.90409435e-09 ... 4.59242209e-08
         6.67120226e-09 2.70914686e-08]
        [2.98220248e-09 2.42440842e-11 5.30401001e-11 ... 2.40147763e-10
         5.26548423e-11 4.86157989e-11]
        [4.99845321e-09 5.55512129e-11 2.71726586e-10 ... 2.09797887e-10
         3.31695019e-11 1.74725984e-10]]
    
       [[4.18021102e-07 1.80844806e-09 4.75171156e-08 ... 5.71578802e-08
         5.59058799e-09 1.54872755e-08]
        [1.12776455e-07 1.01297704e-09 5.03062791e-09 ... 3.08809867e-08
         1.68966108e-09 5.56330138e-09]
        [2.92425767e-07 3.48686835e-09 1.24164314e-08 ... 4.91556342e-08
         4.75869921e-09 3.11101012e-08]
        [3.28792132e-11 1.31608971e-12 5.50265042e-12 ... 1.35834642e-11
         2.83679829e-12 3.38669153e-12]
        [1.74149362e-10 6.32736962e-12 2.51248865e-11 ... 1.96185498e-11
         4.03953000e-12 1.84717051e-11]]
    
       [[3.31152648e-07 1.52582114e-09 5.24846087e-08 ... 2.44688003e-08
         3.89956778e-09 1.96658956e-08]
        [1.42083502e-06 7.94139154e-09 6.34254320e-08 ... 2.20290843e-07
         1.34315119e-08 8.24242150e-08]
        [3.87586056e-07 7.71992426e-09 3.61329739e-08 ... 9.71337428e-08
         1.20081500e-08 1.65051773e-07]
        [1.71313227e-10 6.52739711e-12 4.61270189e-11 ... 6.61518895e-11
         2.07817149e-11 4.30459383e-11]
        [7.12737258e-10 5.73580211e-11 1.42100318e-10 ... 2.00601424e-10
         3.46627796e-11 2.49162024e-10]]]
    
    
      ...
    
    
      [[[3.18335583e-06 5.51743540e-07 1.61668845e-06 ... 2.38412206e-08
         1.04233679e-08 3.76033817e-08]
        [1.67000017e-05 1.71050613e-06 1.03853836e-05 ... 5.07703675e-08
         1.60329332e-08 1.31612225e-07]
        [2.27962937e-06 2.89420484e-07 1.82091856e-06 ... 2.07819504e-08
         6.99454716e-09 3.46180258e-08]
        [4.71832406e-10 1.19577737e-09 8.15324430e-09 ... 4.33208608e-11
         8.98205815e-11 2.89424540e-10]
        [2.40094361e-10 4.84573548e-10 1.44266221e-09 ... 2.41350689e-11
         4.72655734e-11 1.29611100e-10]]
    
       [[5.44495693e-09 2.12661533e-09 6.69881839e-09 ... 2.48732535e-10
         2.68226663e-10 5.20891608e-10]
        [1.44538452e-08 1.34589460e-08 4.46773321e-08 ... 1.48941326e-09
         8.87085128e-10 1.49725898e-09]
        [5.08924757e-07 1.75288719e-07 8.84277654e-07 ... 1.13920180e-08
         1.32162752e-08 3.47067335e-08]
        [3.09850617e-11 4.36071630e-11 3.72556985e-10 ... 1.38931098e-12
         3.13483592e-12 9.11272273e-12]
        [4.91180430e-11 4.50824031e-11 2.63048722e-10 ... 6.64000213e-13
         1.94541432e-12 6.86190080e-12]]
    
       [[7.93682953e-10 1.39524101e-10 1.84678939e-10 ... 1.29124983e-11
         1.57059556e-11 9.67942566e-12]
        [1.34965639e-09 1.57518276e-10 1.61715763e-09 ... 1.56079264e-11
         8.66311450e-12 6.68548680e-12]
        [6.29264676e-08 1.87084748e-08 2.24289636e-07 ... 2.38053077e-10
         4.69783656e-10 7.54385665e-10]
        [3.67834874e-09 7.90237031e-09 1.24990578e-07 ... 6.30102567e-11
         1.46139503e-10 3.23826493e-10]
        [1.78070961e-10 3.42322698e-10 2.94503844e-09 ... 1.25513868e-12
         5.16920578e-12 1.39017530e-11]]
    
       ...
    
       [[7.22629867e-09 3.51152862e-09 2.37862121e-08 ... 1.36204756e-10
         1.28427893e-10 3.08093689e-10]
        [2.40730476e-08 3.99862827e-08 2.83576984e-08 ... 2.82242923e-10
         1.83050727e-10 4.00525335e-10]
        [5.23779534e-07 5.84944587e-07 1.51075528e-06 ... 1.01512609e-09
         1.63253400e-09 5.45704326e-09]
        [1.10946852e-08 2.15773657e-08 3.98551038e-07 ... 9.35595490e-11
         2.79951923e-10 3.13517518e-10]
        [1.96138199e-08 4.26630820e-09 8.39614174e-08 ... 2.05945243e-11
         5.78589121e-11 1.04607170e-10]]
    
       [[9.76087335e-08 1.28088233e-08 8.73567174e-08 ... 4.72728079e-10
         4.15087048e-10 2.00012984e-09]
        [3.10308678e-07 5.19506074e-08 7.12874240e-08 ... 1.06546960e-09
         2.96255520e-10 1.22273358e-09]
        [1.75157857e-06 1.43770720e-07 2.86316094e-07 ... 2.66652189e-09
         5.55379409e-10 5.24585975e-09]
        [3.85212218e-09 1.49785084e-09 2.29186909e-08 ... 1.94086639e-11
         2.26181209e-11 3.54315639e-11]
        [2.46484677e-09 5.82545456e-10 9.92507498e-09 ... 4.13857230e-12
         6.55067103e-12 1.32848533e-11]]
    
       [[5.54912822e-06 4.90514594e-07 3.03753823e-05 ... 1.82179285e-08
         1.55769087e-08 1.35154776e-08]
        [3.20050895e-05 3.29196655e-06 9.01978274e-05 ... 6.56960992e-08
         4.86192029e-08 5.33731352e-08]
        [5.10555947e-05 1.65187737e-06 1.03131597e-05 ... 9.84183188e-08
         2.04595469e-08 7.47392335e-08]
        [7.14538073e-09 6.75118184e-09 8.24692634e-08 ... 1.63732000e-10
         1.95842148e-10 1.90398697e-10]
        [1.44639616e-08 5.44356604e-09 4.26290541e-08 ... 9.28769908e-11
         1.30861239e-10 2.22859481e-10]]]
    
    
      [[[1.24657527e-05 2.31214159e-07 4.32235356e-06 ... 1.36866760e-07
         3.04887990e-08 2.62257771e-07]
        [2.90443641e-05 2.46014508e-07 6.38620577e-06 ... 1.31129738e-07
         1.94803338e-08 3.84419877e-07]
        [2.41121512e-07 1.02904778e-08 5.23728261e-08 ... 3.02598413e-09
         6.86031010e-10 6.15850970e-09]
        [6.24454988e-10 6.45341613e-10 3.24531446e-09 ... 8.57038399e-11
         1.01847308e-10 5.39177980e-10]
        [2.55693605e-10 3.03689823e-10 7.56071594e-10 ... 4.30154627e-11
         4.76046459e-11 1.84437784e-10]]
    
       [[1.36654990e-08 9.98921723e-10 6.15389251e-09 ... 4.19914881e-10
         2.70016814e-10 6.03564199e-10]
        [1.54682300e-06 4.70490491e-08 2.91072269e-07 ... 2.79580252e-08
         5.67848168e-09 1.58932032e-08]
        [1.51735378e-06 1.31239716e-07 6.06994433e-07 ... 5.11803222e-08
         1.39830583e-08 4.88197678e-08]
        [1.88275007e-10 7.52810175e-11 4.01962602e-10 ... 1.62031204e-11
         1.16381097e-11 3.98973909e-11]
        [1.31745198e-10 5.20471444e-11 2.20569729e-10 ... 8.40862362e-12
         7.86614263e-12 3.04170092e-11]]
    
       [[2.83360335e-09 1.77170181e-10 4.54380422e-10 ... 2.69032314e-11
         4.90520471e-11 4.78493183e-11]
        [1.54812014e-07 4.61788519e-09 1.21092718e-08 ... 7.08376913e-10
         3.29998168e-10 1.19965393e-09]
        [1.28387825e-07 2.08772928e-08 7.49871134e-08 ... 1.38889511e-09
         1.46945656e-09 5.24872412e-09]
        [6.17718499e-08 4.29364206e-08 2.19561613e-07 ... 1.85053506e-09
         2.67466294e-09 9.60635216e-09]
        [4.89840613e-10 2.98512826e-10 1.84010396e-09 ... 1.10868051e-11
         2.02990385e-11 8.02567110e-11]]
    
       ...
    
       [[3.61540771e-07 1.76733892e-07 1.35209277e-07 ... 1.08178710e-09
         1.77383208e-09 6.17274321e-09]
        [5.24796292e-07 5.78815047e-07 1.05077838e-07 ... 1.60277214e-09
         8.49477488e-10 1.91138083e-09]
        [1.43894425e-07 9.43658023e-08 1.47320847e-07 ... 2.71915546e-10
         3.51531498e-10 1.04679632e-09]
        [1.78894368e-07 1.54687285e-07 1.68578208e-06 ... 1.05066278e-09
         2.32152653e-09 2.37202635e-09]
        [8.33575597e-10 2.06503953e-10 2.61897637e-09 ... 2.07798297e-12
         6.00661318e-12 1.13135994e-11]]
    
       [[3.15110782e-07 4.41611299e-08 4.54175591e-07 ... 4.04646844e-10
         3.54761553e-10 1.11351250e-09]
        [1.73119915e-05 1.55781026e-06 2.19171034e-06 ... 9.02674024e-09
         2.78763790e-09 6.08044504e-09]
        [1.47100463e-05 5.84594204e-07 2.72518059e-06 ... 1.38178864e-08
         3.68753628e-09 1.02879500e-08]
        [1.14461800e-08 4.14530676e-09 6.78888199e-08 ... 6.53470542e-11
         8.32706681e-11 8.74612882e-11]
        [6.09612749e-10 1.08558849e-10 2.18381313e-09 ... 1.71051899e-12
         3.71967951e-12 4.44618994e-12]]
    
       [[2.24597215e-06 2.27873301e-07 2.07511166e-05 ... 9.36730249e-09
         1.16836949e-08 1.20056782e-08]
        [3.97209369e-05 3.66071095e-06 5.54917438e-04 ... 1.44993663e-07
         1.87177847e-07 1.23584215e-07]
        [1.29171451e-06 3.27026513e-08 6.72257215e-07 ... 2.54497423e-09
         8.79931072e-10 1.80824122e-09]
        [4.86242735e-09 1.03769260e-09 3.59741925e-08 ... 6.23325280e-11
         6.97733676e-11 8.13493994e-11]
        [1.76785531e-09 3.54114266e-10 3.31713212e-09 ... 2.59625654e-11
         2.42346681e-11 4.45816474e-11]]]
    
    
      [[[3.09004645e-05 2.74074438e-07 2.15446676e-06 ... 9.60481088e-07
         5.97782162e-08 2.47780548e-07]
        [1.72091586e-05 1.24148414e-07 6.36186542e-07 ... 2.98351864e-07
         1.22059136e-08 8.33903329e-08]
        [2.68176564e-07 3.65789954e-09 1.04637756e-08 ... 5.77552184e-09
         3.63491320e-10 2.19125451e-09]
        [3.17860938e-09 2.09879025e-09 5.48962564e-09 ... 7.40354833e-10
         2.70823225e-10 9.97291583e-10]
        [2.59826582e-09 1.66124803e-09 3.86928400e-09 ... 5.55744284e-10
         2.09910631e-10 9.42395162e-10]]
    
       [[3.19558717e-06 1.90117987e-07 1.05736194e-06 ... 1.09845750e-07
         2.32265851e-08 3.67280002e-08]
        [1.63198274e-05 5.36590562e-07 1.86498664e-06 ... 2.38130127e-07
         2.93563485e-08 8.63227854e-08]
        [2.70776006e-08 1.42646539e-09 3.71403908e-09 ... 1.17359000e-09
         1.21715638e-10 4.15100954e-10]
        [1.30183364e-09 5.08634412e-10 2.08559592e-09 ... 1.00296854e-10
         3.28593749e-11 1.10905743e-10]
        [4.95719188e-10 2.14128104e-10 1.19933741e-09 ... 5.49971110e-11
         2.77939633e-11 9.69678018e-11]]
    
       [[7.74517048e-07 1.26233445e-07 2.64797336e-07 ... 3.73230336e-08
         8.43524361e-09 8.54291482e-09]
        [2.26445832e-06 2.80442123e-07 4.32018510e-07 ... 4.54331293e-08
         7.04385661e-09 1.93257979e-08]
        [2.48097276e-09 3.46452700e-10 6.94705626e-10 ... 6.67826558e-11
         2.37070311e-11 7.09934958e-11]
        [1.24705215e-08 1.28397337e-08 4.38247660e-08 ... 6.20488994e-10
         4.78069029e-10 1.57342384e-09]
        [8.50895354e-10 6.60607402e-10 3.92369470e-09 ... 6.39611419e-11
         7.67123517e-11 2.41020232e-10]]
    
       ...
    
       [[6.85469558e-06 2.09226027e-06 1.62034701e-06 ... 4.71412243e-09
         6.06553163e-09 5.24173371e-09]
        [1.17917552e-05 7.30513784e-06 2.02341835e-05 ... 3.37173027e-08
         3.48860070e-08 8.62573657e-09]
        [7.44918482e-09 1.36188349e-09 7.30093985e-09 ... 4.71324750e-11
         3.64983078e-11 3.54002105e-11]
        [8.53331414e-08 1.03534006e-07 1.22180131e-06 ... 1.56751212e-09
         1.56570956e-09 1.70740921e-09]
        [7.33821448e-10 4.87045126e-10 3.89756583e-09 ... 1.18003567e-11
         2.44197145e-11 5.25782612e-11]]
    
       [[1.95735320e-05 1.32307741e-06 8.35879928e-06 ... 2.66632174e-08
         1.19304309e-08 1.17639081e-08]
        [7.93789659e-05 3.03662250e-06 2.57714109e-05 ... 1.05026686e-07
         3.77832556e-08 1.64345781e-08]
        [4.61828513e-08 1.28190125e-09 1.45085179e-08 ... 1.14795617e-10
         5.20438276e-11 5.84877771e-11]
        [2.19044711e-08 1.15334657e-08 2.14618908e-07 ... 2.61094285e-10
         2.65427708e-10 3.03842673e-10]
        [6.90398683e-10 4.46331583e-10 4.75699213e-09 ... 1.25165737e-11
         2.58750330e-11 4.13825085e-11]]
    
       [[1.81891392e-05 5.13989016e-06 4.92431427e-05 ... 3.19228874e-08
         4.06456060e-08 3.76261013e-08]
        [9.35329172e-06 2.20309312e-06 7.63766584e-05 ... 3.47299398e-08
         3.95846520e-08 4.16483097e-08]
        [9.16796239e-08 2.49820364e-09 5.64772478e-08 ... 2.28209382e-10
         1.01327675e-10 1.95396907e-10]
        [3.70944120e-09 2.56533839e-09 7.47973772e-08 ... 5.27828337e-11
         7.74027578e-11 1.16008238e-10]
        [1.39124923e-09 8.00250810e-10 6.09461148e-09 ... 3.20353258e-11
         4.12865263e-11 1.10796476e-10]]]]], shape=(1, 19, 19, 5, 80), dtype=float32)
    ----------
    tf.Tensor(
    [[[[14 14 25 14 14]
       [33 33 25 25 33]
       [ 8 33 14 33 33]
       ...
       [29  9 33 33 33]
       [29  9 33 25 33]
       [74  9 25 25 25]]
    
      [[14 14 14 14 14]
       [14 33 14 14 14]
       [14 33 33 14 25]
       ...
       [29  9 50 50 50]
       [ 9  9 50 50 50]
       [ 9 50 50 50 50]]
    
      [[14 58 58 58 58]
       [14 14 58 14 58]
       [14 33  0 33 58]
       ...
       [74  9 50 50 50]
       [ 9  9 50 50 50]
       [ 9 50 50 50 50]]
    
      ...
    
      [[ 0  0  0  2 13]
       [76 73 73 73 13]
       [14 13 13 13 13]
       ...
       [15 15  2  2  2]
       [39  0  0  2  2]
       [ 2  2  0  2  2]]
    
      [[ 0  0  0 73 13]
       [14  0 73 73 73]
       [14  0 73 73 73]
       ...
       [16 16  2  2  2]
       [ 2  0  0  2  2]
       [ 2  2  0  2  2]]
    
      [[ 0  0  0 73 56]
       [ 0  0  0  2  2]
       [ 0  0  0 73  2]
       ...
       [ 0  2  0  2  2]
       [ 0  0  0  2  2]
       [ 2  2  0  2  2]]]], shape=(1, 19, 19, 5), dtype=int64)
    ----------
    tf.Tensor(
    [[[[3.3761917e-05 5.7240727e-06 1.1605949e-07 9.3476089e-08
        1.5319404e-08]
       [9.0035464e-06 1.9679554e-05 4.0415348e-08 1.5330281e-08
        1.6470899e-09]
       [4.8321413e-06 3.5247701e-06 3.8282741e-09 7.6678631e-08
        6.1750038e-10]
       ...
       [3.1560845e-05 4.2289544e-06 3.7189174e-09 7.6779392e-09
        2.1376461e-09]
       [5.1530697e-06 1.6507289e-06 8.1801836e-09 2.0506392e-09
        7.0760287e-10]
       [1.6265214e-05 7.4995573e-06 2.0023119e-08 4.8183604e-09
        1.8289843e-09]]
    
      [[2.5399355e-05 1.4747632e-06 8.3821988e-08 1.7907919e-08
        1.4850645e-08]
       [3.0416987e-07 7.1762241e-07 1.4846525e-07 1.4010579e-09
        3.0369238e-10]
       [4.6222175e-07 2.2191956e-07 1.3381209e-08 1.9720005e-08
        4.2575837e-10]
       ...
       [2.0570326e-06 3.6183367e-06 3.6835533e-07 1.7856330e-08
        6.3393877e-09]
       [7.8273428e-07 2.9488490e-06 1.8091275e-07 1.2228539e-09
        1.9002304e-09]
       [6.3180923e-06 1.8752085e-05 1.1339630e-07 6.8735040e-10
        1.6287177e-09]]
    
      [[8.0894506e-06 1.3425509e-05 2.0783993e-06 9.2803374e-09
        4.3096140e-08]
       [1.7291870e-07 2.6388000e-07 6.3244318e-07 3.6435968e-10
        2.1170841e-09]
       [3.2930274e-07 2.0512032e-07 1.9856996e-07 4.3791277e-09
        1.6587663e-09]
       ...
       [2.1966739e-06 5.1818017e-07 7.5732817e-07 7.5360740e-09
        6.7968329e-09]
       [5.8182754e-06 4.7103211e-07 1.0009649e-06 5.5006411e-10
        7.9099977e-10]
       [1.2983137e-06 2.3526020e-06 2.1014921e-06 1.2898451e-09
        2.4277682e-09]]
    
      ...
    
      [[3.1833558e-06 1.6700002e-05 2.2796294e-06 8.1532443e-09
        1.5816185e-09]
       [1.1290743e-08 1.1996408e-07 3.9930924e-06 7.2022571e-10
        4.4097784e-10]
       [1.2540781e-09 3.7834673e-09 1.0277550e-06 3.2808347e-07
        7.9278166e-09]
       ...
       [1.0695867e-07 2.9010036e-07 1.5107553e-06 3.9855104e-07
        8.3961417e-08]
       [2.6297533e-07 3.1030868e-07 1.7515786e-06 2.2918691e-08
        9.9250750e-09]
       [3.0375382e-05 9.0197827e-05 5.1055595e-05 8.2469263e-08
        4.2629054e-08]]
    
      [[1.2465753e-05 2.9044364e-05 2.4112151e-07 3.3011709e-09
        7.9501544e-10]
       [2.4683018e-08 1.5468230e-06 3.5835667e-06 1.7812338e-09
        1.0075227e-09]
       [4.5731876e-09 1.5481201e-07 1.1051965e-06 1.5293218e-06
        4.3558788e-09]
       ...
       [5.0053802e-07 1.0332320e-06 1.4732085e-07 1.6857821e-06
        2.6189764e-09]
       [4.5417559e-07 1.7311992e-05 1.4710046e-05 6.7888820e-08
        2.1838131e-09]
       [2.0751117e-05 5.5491744e-04 1.2917145e-06 3.5974193e-08
        3.3171321e-09]]
    
      [[3.0900464e-05 1.7209159e-05 2.6817656e-07 9.2009973e-09
        5.9934844e-09]
       [3.1955872e-06 1.6319827e-05 2.7077601e-08 2.0855959e-09
        1.1993374e-09]
       [7.7451705e-07 2.2644583e-06 2.4809728e-09 5.6703605e-08
        3.9236947e-09]
       ...
       [6.8546956e-06 2.0234183e-05 7.4491848e-09 1.2218013e-06
        3.8975658e-09]
       [1.9573532e-05 7.9378966e-05 4.6182851e-08 2.1461891e-07
        4.7569921e-09]
       [4.9243143e-05 7.6376658e-05 9.1679624e-08 7.4797377e-08
        6.0946115e-09]]]], shape=(1, 19, 19, 5), dtype=float32)
    ------------------------------
    ******************************
    (20,)
    (20, 4)
    (20,)
    ------------------------------
    tf.Tensor(
    [0.35717058 0.3244496  0.3722607  0.6702379  0.44434837 0.33506122
     0.50317585 0.664843   0.42050216 0.6194355  0.60223454 0.5219031
     0.74370307 0.5484981  0.7958998  0.35426494 0.3428506  0.8912789
     0.32721937 0.69599354], shape=(20,), dtype=float32)
    ----------
    tf.Tensor(
    [[ 2.7059743e-01  5.3175300e-01  2.9777291e-01  5.4099137e-01]
     [ 3.4971851e-01  4.4791582e-01  4.3337631e-01  5.2286267e-01]
     [ 3.7945616e-01  7.5407481e-01  4.0601790e-01  7.9872346e-01]
     [ 3.6977822e-01  3.5486296e-03  5.6554240e-01  1.7207265e-01]
     [ 4.1144487e-01  2.6267368e-01  4.6499136e-01  2.9498237e-01]
     [ 4.0895405e-01  3.2032228e-01  4.5059934e-01  3.5686040e-01]
     [ 3.9355427e-01  5.3905410e-01  4.8365432e-01  6.0310477e-01]
     [ 3.8767484e-01  5.5146706e-01  4.8659894e-01  6.1367333e-01]
     [ 3.9578706e-01  6.0135430e-01  5.2423489e-01  7.2736353e-01]
     [ 3.8023716e-01  5.8993512e-01  5.5260390e-01  7.3679727e-01]
     [ 3.9626160e-01  7.2257280e-01  5.1905477e-01  8.1632531e-01]
     [ 3.5879594e-01 -3.1143427e-04  6.1345202e-01  1.6304523e-01]
     [ 4.2031127e-01  1.2390946e-01  6.1088568e-01  2.7056772e-01]
     [ 4.0776724e-01  5.8883470e-01  5.5665040e-01  7.3595995e-01]
     [ 3.9167869e-01  5.9490222e-01  5.7173294e-01  7.3629183e-01]
     [ 4.3087074e-01  1.1859395e-01  6.2861335e-01  2.6838174e-01]
     [ 3.8259095e-01  2.6256123e-01  8.6304563e-01  6.0447681e-01]
     [ 4.1633746e-01  2.8635940e-01  8.9989424e-01  5.8211613e-01]
     [ 4.6812421e-01  7.6966703e-01  9.5640236e-01  9.8623025e-01]
     [ 4.4942430e-01  7.4000591e-01  9.7873008e-01  1.0059663e+00]], shape=(20, 4), dtype=float32)
    ----------
    tf.Tensor([9 2 2 5 2 2 2 2 2 2 2 5 2 2 2 2 2 2 2 2], shape=(20,), dtype=int64)
    ------------------------------
    Found 10 boxes for images/test.jpg
    car 0.89 (367, 300) (745, 648)
    car 0.80 (761, 282) (942, 412)
    car 0.74 (159, 303) (346, 440)
    car 0.70 (947, 324) (1280, 705)
    bus 0.67 (5, 266) (220, 407)
    car 0.66 (706, 279) (786, 350)
    car 0.60 (925, 285) (1045, 374)
    car 0.44 (336, 296) (378, 335)
    car 0.37 (965, 273) (1022, 292)
    traffic light 0.36 (681, 195) (692, 214)



![png](output_45_1.png)


**Expected Output**:

<table>
    <tr>
        <td>
            <b>Found 10 boxes for images/test.jpg</b>
        </td>
    </tr>
    <tr>
        <td>
            <b>car</b>
        </td>
        <td>
           0.89 (367, 300) (745, 648)
        </td>
    </tr>
    <tr>
        <td>
            <b>car</b>
        </td>
        <td>
           0.80 (761, 282) (942, 412)
        </td>
    </tr>
    <tr>
        <td>
            <b>car</b>
        </td>
        <td>
           0.74 (159, 303) (346, 440)
        </td>
    </tr>
    <tr>
        <td>
            <b>car</b>
        </td>
        <td>
          0.70 (947, 324) (1280, 705)
        </td>
    </tr>
    <tr>
        <td>
            <b>bus</b>
        </td>
        <td>
           0.67 (5, 266) (220, 407)
        </td>
    </tr>
    <tr>
        <td>
            <b>car</b>
        </td>
        <td>
           0.66 (706, 279) (786, 350)
        </td>
    </tr>
    <tr>
        <td>
            <b>car</b>
        </td>
        <td>
           0.60 (925, 285) (1045, 374)
        </td>
    </tr>
        <tr>
        <td>
            <b>car</b>
        </td>
        <td>
           0.44 (336, 296) (378, 335)
        </td>
    </tr>
    <tr>
        <td>
            <b>car</b>
        </td>
        <td>
           0.37 (965, 273) (1022, 292)
        </td>
    </tr>
    <tr>
        <td>
            <b>traffic light</b>
        </td>
        <td>
           00.36 (681, 195) (692, 214)
        </td>
    </tr>
</table>

The model you've just run is actually able to detect 80 different classes listed in "coco_classes.txt". To test the model on your own images:
    1. Click on "File" in the upper bar of this notebook, then click "Open" to go on your Coursera Hub.
    2. Add your image to this Jupyter Notebook's directory, in the "images" folder
    3. Write your image's name in the cell above code
    4. Run the code and see the output of the algorithm!

If you were to run your session in a for loop over all your images. Here's what you would get:

<center>
<video width="400" height="200" src="nb_images/pred_video_compressed2.mp4" type="video/mp4" controls>
</video>
</center>

<caption><center> Predictions of the YOLO model on pictures taken from a camera while driving around the Silicon Valley <br> Thanks to <a href="https://www.drive.ai/">drive.ai</a> for providing this dataset! </center></caption>

<a name='4'></a>
## 4 - Summary for YOLO

- Input image (608, 608, 3)
- The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output. 
- After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
    - Each cell in a 19x19 grid over the input image gives 425 numbers. 
    - 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture. 
    - 85 = 5 + 80 where 5 is because $(p_c, b_x, b_y, b_h, b_w)$ has 5 numbers, and 80 is the number of classes we'd like to detect
- You then select only few boxes based on:
    - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
    - Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
- This gives you YOLO's final output. 

<font color='blue'>
    
**What you should remember**:
    
- YOLO is a state-of-the-art object detection model that is fast and accurate
- It runs an input image through a CNN, which outputs a 19x19x5x85 dimensional volume. 
- The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
- You filter through all the boxes using non-max suppression. Specifically: 
    - Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
    - Intersection over Union (IoU) thresholding to eliminate overlapping boxes
- Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, previously trained model parameters were used in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise. 

**Congratulations!** You've come to the end of this assignment. 

Here's a quick recap of all you've accomplished.

You've: 

- Detected objects in a car detection dataset
- Implemented non-max suppression to achieve better accuracy
- Implemented intersection over union as a function of NMS
- Created usable bounding box tensors from the model's predictions

Amazing work! If you'd like to know more about the origins of these ideas, spend some time on the papers referenced below. 

<a name='5'></a>
## 5 - References

The ideas presented in this notebook came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's GitHub repository. The pre-trained weights used in this exercise came from the official YOLO website. 
- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
- Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
- Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
- The official YOLO website (https://pjreddie.com/darknet/yolo/) 

### Car detection dataset

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">The Drive.ai Sample Dataset</span> (provided by drive.ai) is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. Thanks to Brody Huval, Chih Hu and Rahul Patel for  providing this data. 
