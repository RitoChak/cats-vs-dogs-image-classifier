# Cats vs Dogs image classifier

## Objective
The objective of this project is to develop an image classifier that can distinguish between images of dogs and cats. The model is trained on a dataset of dog and cat images and evaluated on a separate test set to assess its accuracy.

## Method
### Data Preparation
1. **Dataset**: The dataset is organized into training and testing sets acquired from Kagge, the [link to dataset is here](https://www.kaggle.com/datasets/salader/dogs-vs-cats), each containing images of dogs and cats. It is loaded using TensorFlow's `image_dataset_from_directory` utility.
2. **Normalization**: Images are normalized by dividing pixel values by 255 to bring them into the range [0, 1].

### Convolutional Neural Network (CNN) Architecture
The image classifier is built using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras. The model architecture consists of several convolutional layers with batch normalization, max-pooling layers, and fully connected layers.
- Convolutional Layers: Detect patterns and features in the images.
- Batch Normalization: Normalize and stabilize activations.
- Max-Pooling Layers: Downsample the spatial dimensions.
- Fully Connected Layers: Make predictions based on learned features.

### Model Compilation and Training
The model is compiled using the Adam optimizer and binary crossentropy loss function. It is trained for 10 epochs on the training set, and the validation set is used to monitor the model's performance during training.

## Results
The training and validation accuracy and loss are plotted using matplotlib to visualize the model's learning progress. The model's performance can be assessed based on these plots, helping to identify potential overfitting or underfitting.
