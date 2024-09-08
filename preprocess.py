import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


dataset_path = 'data/fer2013.csv'
# put it in a table
data = pd.read_csv(dataset_path)

images = []
labels=[]

#split images and labels
for i in range(len(data)):
    #i is for each row or each image
    #we have pixels seperated by " " space so we make a list whenever we have a spaceits a new element haha 
    pixels = data['pixels'][i].split(' ')
    # from 120 to 120.
    img = np.array(pixels,dtype='float32')
    img=img.reshape(48,48)
    img = cv2.resize(img,(48,48))
    img = np.stack((img,)*3, axis = -1)
    images.append(img)
    labels.append(data["emotion"][i])
    
images = np.array(images)
labels = np.array(labels)

images = images/255.0

X_train,X_test,Y_train,Y_test= train_test_split(images,labels,test_size=0.2,random_state=123)
   
np.save("data/X_train.npy",X_train)
np.save("data/X_test.npy",X_test) 
np.save("data/Y_train.npy",Y_train) 
np.save("data/Y_test.npy",Y_test) 

 





