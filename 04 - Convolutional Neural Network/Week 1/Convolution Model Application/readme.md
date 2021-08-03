# Convolutional Neural Networks: Application

Welcome to Course 4's second assignment! In this notebook, you will:

- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API

**After this assignment you will be able to:**

- Build and train a ConvNet in TensorFlow for a __binary__ classification problem
- Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
- Explain different use cases for the Sequential and Functional APIs

To complete this assignment, you should already be familiar with TensorFlow. If you are not, please refer back to the **TensorFlow Tutorial** of the third week of Course 2 ("**Improving deep neural networks**").

## Table of Contents

- [1 - Packages](#1)
    - [1.1 - Load the Data and Split the Data into Train/Test Sets](#1-1)
- [2 - Layers in TF Keras](#2)
- [3 - The Sequential API](#3)
    - [3.1 - Create the Sequential Model](#3-1)
        - [Exercise 1 - happyModel](#ex-1)
    - [3.2 - Train and Evaluate the Model](#3-2)
- [4 - The Functional API](#4)
    - [4.1 - Load the SIGNS Dataset](#4-1)
    - [4.2 - Split the Data into Train/Test Sets](#4-2)
    - [4.3 - Forward Propagation](#4-3)
        - [Exercise 2 - convolutional_model](#ex-2)
    - [4.4 - Train the Model](#4-4)
- [5 - History Object](#5)
- [6 - Bibliography](#6)

<a name='1'></a>
## 1 - Packages

As usual, begin by loading in the packages.


```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

%matplotlib inline
np.random.seed(1)
```

<a name='1-1'></a>
### 1.1 - Load the Data and Split the Data into Train/Test Sets

You'll be using the Happy House dataset for this part of the assignment, which contains images of peoples' faces. Your task will be to build a ConvNet that determines whether the people in the images are smiling or not -- because they only get to enter the house if they're smiling!  


```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 600
    number of test examples = 150
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 1)


You can display the images contained in the dataset. Images are **64x64** pixels in RGB format (3 channels).


```python
index = 5
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()
```


![png](output_7_0.png)


<a name='2'></a>
## 2 - Layers in TF Keras 

In the previous assignment, you created layers manually in numpy. In TF Keras, you don't have to write code directly to create layers. Rather, TF Keras has pre-defined layers you can use. 

When you create a layer in TF Keras, you are creating a function that takes some input and transforms it into an output you can reuse later. Nice and easy! 

<a name='3'></a>
## 3 - The Sequential API

In the previous assignment, you built helper functions using `numpy` to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call. Keras is a high-level abstraction built on top of TensorFlow, which allows for even more simplified and optimized model creation and training. 

For the first part of this assignment, you'll create a model using TF Keras' Sequential API, which allows you to build layer by layer, and is ideal for building models where each layer has **exactly one** input tensor and **one** output tensor. 

As you'll see, using the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks. Later in this notebook you'll spend some time building with a more flexible, powerful alternative: the Functional API. 
 

<a name='3-1'></a>
### 3.1 - Create the Sequential Model

As mentioned earlier, the TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order. 

You can also add layers incrementally to a Sequential model with the `.add()` method, or remove them using the `.pop()` method, much like you would in a regular Python list.

Actually, you can think of a Sequential model as behaving like a list of layers. Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.  If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn't be the right choice!

For any layer construction in Keras, you'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. Sequential models can be created by passing a list of layers to the Sequential constructor, like you will do in the next assignment.

<a name='ex-1'></a>
### Exercise 1 - happyModel

Implement the `happyModel` function below to build the following model: `ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Take help from [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 

Also, plug in the following parameters for all the steps:

 - [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): padding 3, input shape 64 x 64 x 3
 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 32 7x7 filters, stride 1
 - [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): for axis 3
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Using default parameters
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 1 neuron and a sigmoid activation. 
 
 
 **Hint:**
 
 Use **tfl** as shorthand for **tensorflow.keras.layers**


```python
# GRADED FUNCTION: happyModel

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tfl.ZeroPadding2D(input_shape=(64, 64, 3) ,padding=3),

            ## Conv2D with 32 7x7 filters and stride of 1
            tfl.Conv2D(32, (7, 7)),

            ## BatchNormalization for axis 3
            tfl.BatchNormalization(axis=3),
            
            ## ReLU
            tfl.ReLU(),
        
            ## Max Pooling 2D with default parameters
            tfl.MaxPooling2D(),
            
            ## Flatten layer
            tfl.Flatten(),

            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tfl.Dense(1, activation='sigmoid')
        ])
    
    return model
```


```python
happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)
    
output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
            ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
            ['BatchNormalization', (None, 64, 64, 32), 128],
            ['ReLU', (None, 64, 64, 32), 0],
            ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
            ['Flatten', (None, 32768), 0],
            ['Dense', (None, 1), 32769, 'sigmoid']]
    
comparator(summary(happy_model), output)
```

    ['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
    ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
    ['BatchNormalization', (None, 64, 64, 32), 128]
    ['ReLU', (None, 64, 64, 32), 0]
    ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
    ['Flatten', (None, 32768), 0]
    ['Dense', (None, 1), 32769, 'sigmoid']
    [32mAll tests passed![0m


Now that your model is created, you can compile it for training with an optimizer and loss of your choice. When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier! If you'd like to read more on how the compiler operates, check the docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).


```python
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

It's time to check your model's parameters with the `.summary()` method. This will display the types of layers you have, the shape of the outputs, and how many parameters are in each layer. 


```python
happy_model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    zero_padding2d_3 (ZeroPaddin (None, 70, 70, 3)         0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 64, 64, 32)        4736      
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 64, 64, 32)        128       
    _________________________________________________________________
    re_lu_3 (ReLU)               (None, 64, 64, 32)        0         
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 32, 32, 32)        0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 32768)             0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 32769     
    =================================================================
    Total params: 37,633
    Trainable params: 37,569
    Non-trainable params: 64
    _________________________________________________________________


<a name='3-2'></a>
### 3.2 - Train and Evaluate the Model

After creating the model, compiling it with your choice of optimizer and loss function, and doing a sanity check on its contents, you are now ready to build! 

Simply call `.fit()` to train. That's it! No need for mini-batching, saving, or complex backpropagation computations. That's all been done for you, as you're using a TensorFlow dataset with the batches specified already. You do have the option to specify epoch number or minibatch size if you like (for example, in the case of an un-batched dataset).


```python
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
```

    Epoch 1/10
    38/38 [==============================] - 4s 94ms/step - loss: 1.2556 - accuracy: 0.6633
    Epoch 2/10
    38/38 [==============================] - 3s 89ms/step - loss: 0.4088 - accuracy: 0.8483
    Epoch 3/10
    38/38 [==============================] - 3s 87ms/step - loss: 0.2393 - accuracy: 0.9050
    Epoch 4/10
    38/38 [==============================] - 3s 90ms/step - loss: 0.1258 - accuracy: 0.9483
    Epoch 5/10
    38/38 [==============================] - 3s 87ms/step - loss: 0.1094 - accuracy: 0.9600
    Epoch 6/10
    38/38 [==============================] - 3s 89ms/step - loss: 0.1150 - accuracy: 0.9500
    Epoch 7/10
    38/38 [==============================] - 3s 90ms/step - loss: 0.0736 - accuracy: 0.9750
    Epoch 8/10
    38/38 [==============================] - 3s 89ms/step - loss: 0.1010 - accuracy: 0.9667
    Epoch 9/10
    38/38 [==============================] - 3s 87ms/step - loss: 0.0982 - accuracy: 0.9567
    Epoch 10/10
    38/38 [==============================] - 3s 87ms/step - loss: 0.1090 - accuracy: 0.9633





    <tensorflow.python.keras.callbacks.History at 0x7fc95a8b6050>



After that completes, just use `.evaluate()` to evaluate against your test set. This function will print the value of the loss function and the performance metrics specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.


```python
happy_model.evaluate(X_test, Y_test)
```

    5/5 [==============================] - 0s 29ms/step - loss: 0.1629 - accuracy: 0.9267





    [0.16294024884700775, 0.9266666769981384]



Easy, right? But what if you need to build a model with shared layers, branches, or multiple inputs and outputs? This is where Sequential, with its beautifully simple yet limited functionality, won't be able to help you. 

Next up: Enter the Functional API, your slightly more complex, highly flexible friend.  

<a name='4'></a>
## 4 - The Functional API

Welcome to the second half of the assignment, where you'll use Keras' flexible [Functional API](https://www.tensorflow.org/guide/keras/functional) to build a ConvNet that can differentiate between 6 sign language digits. 

The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs. Imagine that, where the Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows much more flexibility. Where Sequential is a straight line, a Functional model is a graph, where the nodes of the layers can connect in many more ways than one. 

In the visual example below, the one possible direction of the movement Sequential model is shown in contrast to a skip connection, which is just one of the many ways a Functional model can be constructed. A skip connection, as you might have guessed, skips some layer in the network and feeds the output to a later layer in the network. Don't worry, you'll be spending more time with skip connections very soon! 

<img src="images/seq_vs_func.png" style="width:350px;height:200px;">

<a name='4-1'></a>
### 4.1 - Load the SIGNS Dataset

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.


```python
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
```

<img src="images/SIGNS.png" style="width:800px;height:300px;">

The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples. 


```python
# Example of an image from the dataset
index = 94
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
```

    y = 1



![png](output_28_1.png)


<a name='4-2'></a>
### 4.2 - Split the Data into Train/Test Sets

In Course 2, you built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.

To get started, let's examine the shapes of your data. 


```python
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 1080
    number of test examples = 120
    X_train shape: (1080, 64, 64, 3)
    Y_train shape: (1080, 6)
    X_test shape: (120, 64, 64, 3)
    Y_test shape: (120, 6)


<a name='4-3'></a>
### 4.3 - Forward Propagation

In TensorFlow, there are built-in functions that implement the convolution steps for you. By now, you should be familiar with how TensorFlow builds computational graphs. In the [Functional API](https://www.tensorflow.org/guide/keras/functional), you create a graph of layers. This is what allows such great flexibility.

However, the following model could also be defined using the Sequential API since the information flow is on a single line. But don't deviate. What we want you to learn is to use the functional API.

Begin building your graph of layers by creating an input node that functions as a callable object:

- **input_img = tf.keras.Input(shape=input_shape):** 

Then, create a new node in the graph of layers by calling a layer on the `input_img` object: 

- **tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img):** Read the full documentation on [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).

- **tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'):** `MaxPool2D()` downsamples your input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window.  For max pooling, you usually operate on a single example at a time and a single channel at a time. Read the full documentation on [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).

- **tf.keras.layers.ReLU():** computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU).

- **tf.keras.layers.Flatten()**: given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.  

    * If a tensor P has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), where $k=h \times w \times c$.  "k" equals the product of all the dimension sizes other than the first dimension.
    
    * For example, given a tensor with dimensions [100, 2, 3, 4], it flattens the tensor to be of shape [100, 24], where 24 = 2 * 3 * 4.  You can read the full documentation on [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten).

- **tf.keras.layers.Dense(units= ... , activation='softmax')(F):** given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

In the last function above (`tf.keras.layers.Dense()`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Lastly, before creating the model, you'll need to define the output using the last of the function's compositions (in this example, a Dense layer): 

- **outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)**


#### Window, kernel, filter, pool

The words "kernel" and "filter" are used to refer to the same thing. The word "filter" accounts for the amount of "kernels" that will be used in a single convolution layer. "Pool" is the name of the operation that takes the max or average value of the kernels. 

This is why the parameter `pool_size` refers to `kernel_size`, and you use `(f,f)` to refer to the filter size. 

Pool size and kernel size refer to the same thing in different objects - They refer to the shape of the window where the operation takes place. 

<a name='ex-2'></a>
### Exercise 2 - convolutional_model

Implement the `convolutional_model` function below to build the following model: `CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Use the functions above! 

Also, plug in the following parameters for all the steps:

 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 8 4 by 4 filters, stride 1, padding is "SAME"
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - **Conv2D**: Use 16 2 by 2 filters, stride 1, padding is "SAME"
 - **ReLU**
 - **MaxPool2D**: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 6 neurons and a softmax activation. 


```python
# GRADED FUNCTION: convolutional_model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tf.keras.layers.Conv2D(8, (4, 4), padding='same')(input_img)
    ## RELU
    A1 = tf.keras.layers.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.keras.layers.MaxPooling2D((8, 8), 8, padding='same')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tf.keras.layers.Conv2D(16, (2, 2), padding='same')(P1)
    ## RELU
    A2 = tf.keras.layers.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.keras.layers.MaxPooling2D((4, 4), 4, padding='same')(A2)
    ## FLATTEN
    F = tf.keras.layers.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tf.keras.layers.Dense(6, activation='softmax')(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
```


```python
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)
```

    Model: "functional_7"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_5 (InputLayer)         [(None, 64, 64, 3)]       0         
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 64, 64, 8)         392       
    _________________________________________________________________
    re_lu_10 (ReLU)              (None, 64, 64, 8)         0         
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling (None, 8, 8, 8)           0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 8, 8, 16)          528       
    _________________________________________________________________
    re_lu_11 (ReLU)              (None, 8, 8, 16)          0         
    _________________________________________________________________
    max_pooling2d_11 (MaxPooling (None, 2, 2, 16)          0         
    _________________________________________________________________
    flatten_7 (Flatten)          (None, 64)                0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 6)                 390       
    =================================================================
    Total params: 1,310
    Trainable params: 1,310
    Non-trainable params: 0
    _________________________________________________________________
    [32mAll tests passed![0m


Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model! 

<a name='4-4'></a>
### 4.4 - Train the Model


```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
```

    Epoch 1/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.8086 - accuracy: 0.1694 - val_loss: 1.7917 - val_accuracy: 0.2500
    Epoch 2/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7894 - accuracy: 0.2139 - val_loss: 1.7883 - val_accuracy: 0.2250
    Epoch 3/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.7854 - accuracy: 0.2213 - val_loss: 1.7860 - val_accuracy: 0.2583
    Epoch 4/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.7821 - accuracy: 0.2500 - val_loss: 1.7835 - val_accuracy: 0.2583
    Epoch 5/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.7786 - accuracy: 0.2620 - val_loss: 1.7802 - val_accuracy: 0.2667
    Epoch 6/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7741 - accuracy: 0.3056 - val_loss: 1.7757 - val_accuracy: 0.2500
    Epoch 7/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7667 - accuracy: 0.3426 - val_loss: 1.7685 - val_accuracy: 0.3417
    Epoch 8/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.7582 - accuracy: 0.3648 - val_loss: 1.7600 - val_accuracy: 0.4083
    Epoch 9/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.7466 - accuracy: 0.3981 - val_loss: 1.7469 - val_accuracy: 0.3917
    Epoch 10/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7294 - accuracy: 0.4130 - val_loss: 1.7289 - val_accuracy: 0.4000
    Epoch 11/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7039 - accuracy: 0.4296 - val_loss: 1.7021 - val_accuracy: 0.4583
    Epoch 12/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.6686 - accuracy: 0.4602 - val_loss: 1.6673 - val_accuracy: 0.4667
    Epoch 13/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.6255 - accuracy: 0.4741 - val_loss: 1.6241 - val_accuracy: 0.4667
    Epoch 14/100
    17/17 [==============================] - 2s 94ms/step - loss: 1.5752 - accuracy: 0.4944 - val_loss: 1.5752 - val_accuracy: 0.4583
    Epoch 15/100
    17/17 [==============================] - 2s 94ms/step - loss: 1.5198 - accuracy: 0.5019 - val_loss: 1.5236 - val_accuracy: 0.5083
    Epoch 16/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.4644 - accuracy: 0.5102 - val_loss: 1.4745 - val_accuracy: 0.5167
    Epoch 17/100
    17/17 [==============================] - 2s 94ms/step - loss: 1.4107 - accuracy: 0.5250 - val_loss: 1.4232 - val_accuracy: 0.5333
    Epoch 18/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.3610 - accuracy: 0.5370 - val_loss: 1.3763 - val_accuracy: 0.5250
    Epoch 19/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.3157 - accuracy: 0.5500 - val_loss: 1.3327 - val_accuracy: 0.5417
    Epoch 20/100
    17/17 [==============================] - 2s 94ms/step - loss: 1.2718 - accuracy: 0.5667 - val_loss: 1.2910 - val_accuracy: 0.5667
    Epoch 21/100
    17/17 [==============================] - 2s 94ms/step - loss: 1.2325 - accuracy: 0.5880 - val_loss: 1.2536 - val_accuracy: 0.5583
    Epoch 22/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.1947 - accuracy: 0.5981 - val_loss: 1.2159 - val_accuracy: 0.5583
    Epoch 23/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.1616 - accuracy: 0.6056 - val_loss: 1.1819 - val_accuracy: 0.5667
    Epoch 24/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.1302 - accuracy: 0.6204 - val_loss: 1.1472 - val_accuracy: 0.5750
    Epoch 25/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.0986 - accuracy: 0.6269 - val_loss: 1.1154 - val_accuracy: 0.5917
    Epoch 26/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.0689 - accuracy: 0.6417 - val_loss: 1.0865 - val_accuracy: 0.5917
    Epoch 27/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.0413 - accuracy: 0.6463 - val_loss: 1.0606 - val_accuracy: 0.6083
    Epoch 28/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.0176 - accuracy: 0.6528 - val_loss: 1.0353 - val_accuracy: 0.6167
    Epoch 29/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.9930 - accuracy: 0.6620 - val_loss: 1.0125 - val_accuracy: 0.6250
    Epoch 30/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.9690 - accuracy: 0.6731 - val_loss: 0.9897 - val_accuracy: 0.6333
    Epoch 31/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.9466 - accuracy: 0.6843 - val_loss: 0.9679 - val_accuracy: 0.6250
    Epoch 32/100
    17/17 [==============================] - 2s 94ms/step - loss: 0.9255 - accuracy: 0.6907 - val_loss: 0.9474 - val_accuracy: 0.6333
    Epoch 33/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.9059 - accuracy: 0.6944 - val_loss: 0.9284 - val_accuracy: 0.6500
    Epoch 34/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.8865 - accuracy: 0.7028 - val_loss: 0.9106 - val_accuracy: 0.6417
    Epoch 35/100
    17/17 [==============================] - 2s 94ms/step - loss: 0.8680 - accuracy: 0.7093 - val_loss: 0.8932 - val_accuracy: 0.6750
    Epoch 36/100
    17/17 [==============================] - 2s 99ms/step - loss: 0.8502 - accuracy: 0.7213 - val_loss: 0.8758 - val_accuracy: 0.6667
    Epoch 37/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.8338 - accuracy: 0.7204 - val_loss: 0.8599 - val_accuracy: 0.7000
    Epoch 38/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.8174 - accuracy: 0.7315 - val_loss: 0.8449 - val_accuracy: 0.7250
    Epoch 39/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.8021 - accuracy: 0.7389 - val_loss: 0.8301 - val_accuracy: 0.7167
    Epoch 40/100
    17/17 [==============================] - 2s 99ms/step - loss: 0.7870 - accuracy: 0.7472 - val_loss: 0.8169 - val_accuracy: 0.7250
    Epoch 41/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.7733 - accuracy: 0.7528 - val_loss: 0.8038 - val_accuracy: 0.7333
    Epoch 42/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.7592 - accuracy: 0.7593 - val_loss: 0.7916 - val_accuracy: 0.7333
    Epoch 43/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.7464 - accuracy: 0.7648 - val_loss: 0.7809 - val_accuracy: 0.7500
    Epoch 44/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.7338 - accuracy: 0.7741 - val_loss: 0.7694 - val_accuracy: 0.7500
    Epoch 45/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.7212 - accuracy: 0.7796 - val_loss: 0.7584 - val_accuracy: 0.7500
    Epoch 46/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.7100 - accuracy: 0.7824 - val_loss: 0.7484 - val_accuracy: 0.7333
    Epoch 47/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6981 - accuracy: 0.7870 - val_loss: 0.7388 - val_accuracy: 0.7500
    Epoch 48/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6878 - accuracy: 0.7880 - val_loss: 0.7291 - val_accuracy: 0.7500
    Epoch 49/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6768 - accuracy: 0.7926 - val_loss: 0.7205 - val_accuracy: 0.7583
    Epoch 50/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6670 - accuracy: 0.7963 - val_loss: 0.7126 - val_accuracy: 0.7583
    Epoch 51/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6569 - accuracy: 0.8000 - val_loss: 0.7047 - val_accuracy: 0.7583
    Epoch 52/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6478 - accuracy: 0.8056 - val_loss: 0.6975 - val_accuracy: 0.7667
    Epoch 53/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6386 - accuracy: 0.8046 - val_loss: 0.6905 - val_accuracy: 0.7750
    Epoch 54/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6300 - accuracy: 0.8120 - val_loss: 0.6829 - val_accuracy: 0.7750
    Epoch 55/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6210 - accuracy: 0.8148 - val_loss: 0.6773 - val_accuracy: 0.7750
    Epoch 56/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6129 - accuracy: 0.8185 - val_loss: 0.6697 - val_accuracy: 0.7750
    Epoch 57/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6041 - accuracy: 0.8222 - val_loss: 0.6653 - val_accuracy: 0.7667
    Epoch 58/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5969 - accuracy: 0.8204 - val_loss: 0.6578 - val_accuracy: 0.7583
    Epoch 59/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5888 - accuracy: 0.8296 - val_loss: 0.6520 - val_accuracy: 0.7667
    Epoch 60/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5816 - accuracy: 0.8296 - val_loss: 0.6472 - val_accuracy: 0.7583
    Epoch 61/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5730 - accuracy: 0.8324 - val_loss: 0.6416 - val_accuracy: 0.7750
    Epoch 62/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5672 - accuracy: 0.8315 - val_loss: 0.6372 - val_accuracy: 0.7583
    Epoch 63/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.5591 - accuracy: 0.8324 - val_loss: 0.6315 - val_accuracy: 0.7667
    Epoch 64/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.5525 - accuracy: 0.8352 - val_loss: 0.6281 - val_accuracy: 0.7583
    Epoch 65/100
    17/17 [==============================] - 2s 94ms/step - loss: 0.5449 - accuracy: 0.8380 - val_loss: 0.6228 - val_accuracy: 0.7667
    Epoch 66/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.5390 - accuracy: 0.8380 - val_loss: 0.6189 - val_accuracy: 0.7583
    Epoch 67/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5312 - accuracy: 0.8370 - val_loss: 0.6146 - val_accuracy: 0.7750
    Epoch 68/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5259 - accuracy: 0.8426 - val_loss: 0.6118 - val_accuracy: 0.7583
    Epoch 69/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5195 - accuracy: 0.8417 - val_loss: 0.6078 - val_accuracy: 0.7750
    Epoch 70/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5141 - accuracy: 0.8463 - val_loss: 0.6050 - val_accuracy: 0.7667
    Epoch 71/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5077 - accuracy: 0.8463 - val_loss: 0.6001 - val_accuracy: 0.7750
    Epoch 72/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5020 - accuracy: 0.8500 - val_loss: 0.5975 - val_accuracy: 0.7750
    Epoch 73/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.4956 - accuracy: 0.8519 - val_loss: 0.5954 - val_accuracy: 0.7750
    Epoch 74/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4902 - accuracy: 0.8537 - val_loss: 0.5924 - val_accuracy: 0.7833
    Epoch 75/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4846 - accuracy: 0.8565 - val_loss: 0.5904 - val_accuracy: 0.7750
    Epoch 76/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4796 - accuracy: 0.8565 - val_loss: 0.5871 - val_accuracy: 0.7833
    Epoch 77/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4737 - accuracy: 0.8593 - val_loss: 0.5857 - val_accuracy: 0.7833
    Epoch 78/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.4687 - accuracy: 0.8583 - val_loss: 0.5821 - val_accuracy: 0.7833
    Epoch 79/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.4631 - accuracy: 0.8630 - val_loss: 0.5813 - val_accuracy: 0.7833
    Epoch 80/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4586 - accuracy: 0.8630 - val_loss: 0.5761 - val_accuracy: 0.7833
    Epoch 81/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.4533 - accuracy: 0.8694 - val_loss: 0.5760 - val_accuracy: 0.7833
    Epoch 82/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4489 - accuracy: 0.8685 - val_loss: 0.5734 - val_accuracy: 0.7833
    Epoch 83/100
    17/17 [==============================] - 2s 99ms/step - loss: 0.4443 - accuracy: 0.8731 - val_loss: 0.5726 - val_accuracy: 0.7917
    Epoch 84/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4403 - accuracy: 0.8731 - val_loss: 0.5699 - val_accuracy: 0.7833
    Epoch 85/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4349 - accuracy: 0.8778 - val_loss: 0.5705 - val_accuracy: 0.7917
    Epoch 86/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4313 - accuracy: 0.8750 - val_loss: 0.5702 - val_accuracy: 0.7917
    Epoch 87/100
    17/17 [==============================] - 2s 105ms/step - loss: 0.4274 - accuracy: 0.8815 - val_loss: 0.5714 - val_accuracy: 0.8000
    Epoch 88/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4236 - accuracy: 0.8787 - val_loss: 0.5742 - val_accuracy: 0.8000
    Epoch 89/100
    17/17 [==============================] - 2s 99ms/step - loss: 0.4205 - accuracy: 0.8769 - val_loss: 0.5771 - val_accuracy: 0.8000
    Epoch 90/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4175 - accuracy: 0.8824 - val_loss: 0.5783 - val_accuracy: 0.8000
    Epoch 91/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4145 - accuracy: 0.8824 - val_loss: 0.5820 - val_accuracy: 0.8000
    Epoch 92/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4124 - accuracy: 0.8824 - val_loss: 0.5818 - val_accuracy: 0.8083
    Epoch 93/100
    17/17 [==============================] - 2s 105ms/step - loss: 0.4093 - accuracy: 0.8843 - val_loss: 0.5833 - val_accuracy: 0.8083
    Epoch 94/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4069 - accuracy: 0.8824 - val_loss: 0.5865 - val_accuracy: 0.8083
    Epoch 95/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4050 - accuracy: 0.8815 - val_loss: 0.5898 - val_accuracy: 0.8000
    Epoch 96/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.4034 - accuracy: 0.8815 - val_loss: 0.5834 - val_accuracy: 0.8083
    Epoch 97/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4013 - accuracy: 0.8787 - val_loss: 0.5789 - val_accuracy: 0.7917
    Epoch 98/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.3998 - accuracy: 0.8824 - val_loss: 0.5684 - val_accuracy: 0.8167
    Epoch 99/100
    17/17 [==============================] - 2s 99ms/step - loss: 0.3955 - accuracy: 0.8815 - val_loss: 0.5577 - val_accuracy: 0.8083
    Epoch 100/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.3900 - accuracy: 0.8870 - val_loss: 0.5499 - val_accuracy: 0.8083


<a name='5'></a>
## 5 - History Object 

The history object is an output of the `.fit()` operation, and provides a record of all the loss and metric values in memory. It's stored as a dictionary that you can retrieve at `history.history`: 


```python
history.history
```




    {'loss': [1.8085790872573853,
      1.7893632650375366,
      1.7854124307632446,
      1.782131314277649,
      1.7785992622375488,
      1.7740784883499146,
      1.7667232751846313,
      1.7581803798675537,
      1.7466158866882324,
      1.7294142246246338,
      1.7039345502853394,
      1.6686116456985474,
      1.6254606246948242,
      1.5752137899398804,
      1.5198423862457275,
      1.4644064903259277,
      1.4107491970062256,
      1.3610438108444214,
      1.3156901597976685,
      1.2717714309692383,
      1.232542634010315,
      1.194738507270813,
      1.1616305112838745,
      1.130167007446289,
      1.0985965728759766,
      1.0689306259155273,
      1.0412734746932983,
      1.017575740814209,
      0.9930370450019836,
      0.9689539074897766,
      0.9465653300285339,
      0.9254798889160156,
      0.9058670401573181,
      0.8865059614181519,
      0.8679652214050293,
      0.8501831889152527,
      0.8337886333465576,
      0.8174141645431519,
      0.8021332025527954,
      0.7870262265205383,
      0.7733106017112732,
      0.7591623067855835,
      0.7464277744293213,
      0.7337924242019653,
      0.7212186455726624,
      0.7100012898445129,
      0.6981265544891357,
      0.6878299117088318,
      0.6768171787261963,
      0.6670097708702087,
      0.6569427847862244,
      0.6477684378623962,
      0.6386044025421143,
      0.6299691796302795,
      0.6209980249404907,
      0.6128971576690674,
      0.6041393280029297,
      0.5969353318214417,
      0.588762640953064,
      0.581572949886322,
      0.5730383992195129,
      0.5672404766082764,
      0.5591052174568176,
      0.5525025725364685,
      0.5448704361915588,
      0.5389803647994995,
      0.531212568283081,
      0.5259110331535339,
      0.5195028185844421,
      0.5140868425369263,
      0.5076822638511658,
      0.5020172595977783,
      0.495567262172699,
      0.49015650153160095,
      0.4846491515636444,
      0.4795895516872406,
      0.4736737012863159,
      0.4687272906303406,
      0.46305444836616516,
      0.4585707485675812,
      0.4532993733882904,
      0.4488500952720642,
      0.4443112909793854,
      0.4403337240219116,
      0.43494850397109985,
      0.4312644898891449,
      0.42739278078079224,
      0.4235925078392029,
      0.42050066590309143,
      0.41754645109176636,
      0.4145447313785553,
      0.41237279772758484,
      0.4093132019042969,
      0.40694618225097656,
      0.4050498902797699,
      0.4034331738948822,
      0.4013069272041321,
      0.39983001351356506,
      0.3954865634441376,
      0.38998016715049744],
     'accuracy': [0.16944444179534912,
      0.21388888359069824,
      0.2212962955236435,
      0.25,
      0.2620370388031006,
      0.3055555522441864,
      0.34259259700775146,
      0.364814817905426,
      0.39814814925193787,
      0.41296297311782837,
      0.4296296238899231,
      0.4601851999759674,
      0.4740740656852722,
      0.49444442987442017,
      0.5018518567085266,
      0.510185182094574,
      0.5249999761581421,
      0.5370370149612427,
      0.550000011920929,
      0.5666666626930237,
      0.5879629850387573,
      0.5981481671333313,
      0.605555534362793,
      0.6203703880310059,
      0.6268518567085266,
      0.6416666507720947,
      0.6462963223457336,
      0.6527777910232544,
      0.6620370149612427,
      0.6731481552124023,
      0.6842592358589172,
      0.6907407641410828,
      0.6944444179534912,
      0.7027778029441833,
      0.7092592716217041,
      0.7212963104248047,
      0.720370352268219,
      0.7314814925193787,
      0.7388888597488403,
      0.7472222447395325,
      0.7527777552604675,
      0.7592592835426331,
      0.7648147940635681,
      0.7740740776062012,
      0.779629647731781,
      0.7824074029922485,
      0.7870370149612427,
      0.7879629731178284,
      0.7925925850868225,
      0.7962962985038757,
      0.800000011920929,
      0.8055555820465088,
      0.8046296238899231,
      0.8120370507240295,
      0.8148148059844971,
      0.8185185194015503,
      0.8222222328186035,
      0.8203703761100769,
      0.8296296000480652,
      0.8296296000480652,
      0.8324074149131775,
      0.8314814567565918,
      0.8324074149131775,
      0.835185170173645,
      0.8379629850387573,
      0.8379629850387573,
      0.8370370268821716,
      0.8425925970077515,
      0.8416666388511658,
      0.8462963104248047,
      0.8462963104248047,
      0.8500000238418579,
      0.8518518805503845,
      0.8537036776542664,
      0.8564814925193787,
      0.8564814925193787,
      0.8592592477798462,
      0.8583333492279053,
      0.8629629611968994,
      0.8629629611968994,
      0.8694444298744202,
      0.8685185313224792,
      0.8731481432914734,
      0.8731481432914734,
      0.8777777552604675,
      0.875,
      0.8814814686775208,
      0.8787037134170532,
      0.8768518567085266,
      0.8824074268341064,
      0.8824074268341064,
      0.8824074268341064,
      0.8842592835426331,
      0.8824074268341064,
      0.8814814686775208,
      0.8814814686775208,
      0.8787037134170532,
      0.8824074268341064,
      0.8814814686775208,
      0.8870370388031006],
     'val_loss': [1.791674017906189,
      1.7883325815200806,
      1.7860283851623535,
      1.7835191488265991,
      1.780192255973816,
      1.7757278680801392,
      1.7685281038284302,
      1.7600111961364746,
      1.746856927871704,
      1.7288674116134644,
      1.7020834684371948,
      1.6672943830490112,
      1.6241015195846558,
      1.5752094984054565,
      1.523611307144165,
      1.4745219945907593,
      1.423167109489441,
      1.37630295753479,
      1.3327199220657349,
      1.2909855842590332,
      1.2536320686340332,
      1.2158626317977905,
      1.181909203529358,
      1.1471997499465942,
      1.1153571605682373,
      1.086488127708435,
      1.0605616569519043,
      1.0352652072906494,
      1.0124701261520386,
      0.9897065758705139,
      0.967858076095581,
      0.9473969340324402,
      0.9283888339996338,
      0.9105513691902161,
      0.8931896090507507,
      0.8758019208908081,
      0.8598823547363281,
      0.8449307680130005,
      0.8301196694374084,
      0.8168609738349915,
      0.8037670850753784,
      0.791610062122345,
      0.7808576822280884,
      0.7694348692893982,
      0.7583531737327576,
      0.7483829259872437,
      0.7388092279434204,
      0.7291425466537476,
      0.7205237150192261,
      0.7125858664512634,
      0.7047105431556702,
      0.6974714994430542,
      0.6905412077903748,
      0.6829031705856323,
      0.6772955656051636,
      0.6697438359260559,
      0.6653314828872681,
      0.6577872633934021,
      0.6520000696182251,
      0.6471520066261292,
      0.641555666923523,
      0.6371896266937256,
      0.6314704418182373,
      0.6281322240829468,
      0.6228113174438477,
      0.6189084649085999,
      0.6146307587623596,
      0.6117916703224182,
      0.6077700853347778,
      0.6050096154212952,
      0.6001392006874084,
      0.5974693894386292,
      0.5954334139823914,
      0.5923639535903931,
      0.5904408693313599,
      0.5871075987815857,
      0.5857341289520264,
      0.5821104645729065,
      0.5813038945198059,
      0.5761427283287048,
      0.5760385990142822,
      0.5733987092971802,
      0.572590172290802,
      0.5699223279953003,
      0.5704997777938843,
      0.5701801180839539,
      0.5714409947395325,
      0.5741758942604065,
      0.5770864486694336,
      0.578250527381897,
      0.5819970965385437,
      0.5818290710449219,
      0.5832651853561401,
      0.5865427851676941,
      0.589816153049469,
      0.5833712816238403,
      0.5789287090301514,
      0.568448007106781,
      0.5577368140220642,
      0.5498754978179932],
     'val_accuracy': [0.25,
      0.22499999403953552,
      0.25833332538604736,
      0.25833332538604736,
      0.2666666805744171,
      0.25,
      0.34166666865348816,
      0.40833333134651184,
      0.3916666805744171,
      0.4000000059604645,
      0.4583333432674408,
      0.46666666865348816,
      0.46666666865348816,
      0.4583333432674408,
      0.5083333253860474,
      0.5166666507720947,
      0.5333333611488342,
      0.5249999761581421,
      0.5416666865348816,
      0.5666666626930237,
      0.5583333373069763,
      0.5583333373069763,
      0.5666666626930237,
      0.574999988079071,
      0.5916666388511658,
      0.5916666388511658,
      0.6083333492279053,
      0.6166666746139526,
      0.625,
      0.6333333253860474,
      0.625,
      0.6333333253860474,
      0.6499999761581421,
      0.6416666507720947,
      0.675000011920929,
      0.6666666865348816,
      0.699999988079071,
      0.7250000238418579,
      0.7166666388511658,
      0.7250000238418579,
      0.7333333492279053,
      0.7333333492279053,
      0.75,
      0.75,
      0.75,
      0.7333333492279053,
      0.75,
      0.75,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7666666507720947,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7666666507720947,
      0.7583333253860474,
      0.7666666507720947,
      0.7583333253860474,
      0.7749999761581421,
      0.7583333253860474,
      0.7666666507720947,
      0.7583333253860474,
      0.7666666507720947,
      0.7583333253860474,
      0.7749999761581421,
      0.7583333253860474,
      0.7749999761581421,
      0.7666666507720947,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7833333611488342,
      0.7749999761581421,
      0.7833333611488342,
      0.7833333611488342,
      0.7833333611488342,
      0.7833333611488342,
      0.7833333611488342,
      0.7833333611488342,
      0.7833333611488342,
      0.7916666865348816,
      0.7833333611488342,
      0.7916666865348816,
      0.7916666865348816,
      0.800000011920929,
      0.800000011920929,
      0.800000011920929,
      0.800000011920929,
      0.800000011920929,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.800000011920929,
      0.8083333373069763,
      0.7916666865348816,
      0.8166666626930237,
      0.8083333373069763,
      0.8083333373069763]}



Now visualize the loss over time using `history.history`: 


```python
# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on. 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```




    [Text(0, 0.5, 'Accuracy'), Text(0.5, 0, 'Epoch')]




![png](output_41_1.png)



![png](output_41_2.png)


**Congratulations**! You've finished the assignment and built two models: One that recognizes  smiles, and another that recognizes SIGN language with almost 80% accuracy on the test set. In addition to that, you now also understand the applications of two Keras APIs: Sequential and Functional. Nicely done! 

By now, you know a bit about how the Functional API works and may have glimpsed the possibilities. In your next assignment, you'll really get a feel for its power when you get the opportunity to build a very deep ConvNet, using ResNets! 

<a name='6'></a>
## 6 - Bibliography

You're always encouraged to read the official documentation. To that end, you can find the docs for the Sequential and Functional APIs here: 

https://www.tensorflow.org/guide/keras/sequential_model

https://www.tensorflow.org/guide/keras/functional
