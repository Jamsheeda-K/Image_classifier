#Keras runs on GPU, it seems there is some graphic cards installed in my computer, so there is some problem there, 
#so i am assigning it to run only on CPUs. This method reduces the performance  
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# importing all libraries
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import cv2

# I can use the model I saved which is trained on 1457 datapoints 
model_saved = keras.models.load_model('C:/Jamsheeda/Spiced_projects/week_9/Project/imageclassifier/models/mnist_model.h5')
def predict_class(my_image):
    my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
    print(my_image.shape)
    my_image = my_image.reshape(28,28).reshape(-1)
    #pic = keras.preprocessing.image.load_img(my_image,target_size=(224,224))
    #pic.show()
    #np_pic = np.array(pic)
    #batch_pic = np.expand_dims(my_image,axis=0) 
    #processed_pic = keras.applications.mobilenet_v2.preprocess_input(batch_pic)
    y_pred = model_saved.predict(my_image).round(3)
    #categories = ['apple','knife','locher','orange','pencil','pumpkin','tomato']
    print(y_pred)
    max_pred_prob = y_pred[0].max()
    if max_pred_prob >= 0.7:
        return_text = f'I am very certain that it is a {categories[y_pred[0].argmax()].upper()}'
    else:
        return_text = f'I am not very sure, but I guess it is a {categories[y_pred[0].argmax()].upper()}'
    for i in range(len(y_pred[0])):
        if (abs(max_pred_prob-y_pred[0][i]) < 0.1) and (max_pred_prob!=y_pred[0][i]) : 
            return_text = f'My brain says: {categories[y_pred[0].argmax()].upper()} but my gut says: {categories[i].upper()}'

    #for category, index in ['apple','banana']: 
        #if index == category_index:
           # predicted_class = category
    return return_text
