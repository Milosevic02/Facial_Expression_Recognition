import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input,Conv2D,Dense,Flatten,Dropout,MaxPooling2D
from keras.models import Model

data = pd.read_csv('fer2013.csv',header = 0)

train = data.loc[data['Usage'] == 'Training']
test = data.loc[data['Usage'] != 'Training']

X_train = np.array([np.array(pixel_list.split(' ')).astype(float)/255. for pixel_list in train['pixels']])
X_test = np.array([np.array(pixel_list.split(' ')).astype(float)/255. for pixel_list in test['pixels']])

X_train = X_train.reshape(-1,48,48,1)
X_test = X_test.reshape(-1,48,48,1)

y_train = train['emotion']
y_test = test['emotion']