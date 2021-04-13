#Keras runs on GPU, it seems there is some graphic cards installed in my computer, so there is some problem there, 
#so i am assigning it to run only on CPUs. This method reduces the performance  
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# importing all libraries
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt


#Importing one of the pretrainde models from keras > xception
from tensorflow.keras.applications import xception
model = xception.Xception()

img = keras.preprocessing.image.load_img('C:\Jamsheeda\Spiced_projects\week_9\Project\imageclassifier\models\picture.png',target_size=(299,299))

plt.imshow(img)
img = np.array(img)
img = img.reshape((1,299,299,3))

#scales the image such that it has a mean of 0
img = xception.preprocess_input(img)

#probability for each class
img_pred = model.predict(img)
for i in range(5):
    print(xception.decode_predictions(img_pred)[0][i][1:3])