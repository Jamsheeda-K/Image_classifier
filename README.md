# ImageClassifier

Use Neural Networks to automatically distinguish between objects in pictures. 


## Usage

take a picture with the webcam (press `space`) and save them in a folder:

```python
python imageclassifier/capture.py data/test/faces/
```

exit the program with `q`


## Project Goal

Build a machine learning pipeline with Keras (and possibly other tools) that classifies images of objects. 


## Project Tasks

### Data 

- together with the group, collect a data set of images of objects, gestures or facial expressions
- read in and process the images to be used in a machine learning model
    - split the data into a training and validation data-set
    - scale, crop and normalize the images
    - use real-time image augmentation to enlarge the data-set
    
### Models

1. Start with a baseline model (e.g. a Logistic Regression model).
2. Use a fully connected neural network.
3. Add convolutional layers. 
4. Try out a pre-trained network.
5. Use fine-tuning on the pre-trained network.

  
### Evaluation

- save the model to disk 
- use the provided script to classify images from the webcam in real time
- display the classification probabilities on the screen




