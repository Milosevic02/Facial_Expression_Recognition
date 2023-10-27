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

K = len(set(y_train))
print("number of classes: ",K)

print("x_train.shape: " , X_train.shape)
print("y_train.shape",y_train.shape)
input_shape = (48,48,1)
i = Input(shape = input_shape)
x = Conv2D(32,(3,3),activation = 'relu')(i)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64,(3,3),activation = 'relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128,(3,3),activation = 'relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256,(3,3),activation = 'relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128,activation = 'relu')(x)
x = Dropout(0.3)(x)
x = Dense(K,activation = 'softmax')(x)

model = Model(i,x)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

r = model.fit(X_train,y_train,batch_size=32,epochs=10,verbose=1, validation_data=(X_test, y_test))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


p_test = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(7)))

