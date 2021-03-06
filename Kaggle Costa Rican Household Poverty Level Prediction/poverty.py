# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:51:41 2018

@author: david
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csvimport pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

data_all = pd.read_csv('../input/train.csv')
data_all = data_all.fillna(0)
Y = data_all['Target'].values
X = data_all.drop(columns=['v18q1', 'rez_esc', 'idhogar', 'dependency', 'edjefe',
                           'edjefa','meaneduc', 'Target', 'Id'],axis = 1)
Y = Y.astype('float32')
X = X.astype('float32')
scaler = MinMaxScaler(feature_range = (0,1))
X = X.values
X = scaler.fit_transform(X)
Y = pd.get_dummies(Y)
Y = Y.values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)

model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='elu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(4, activation = "softmax"))

model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", 
              metrics=["accuracy"])


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)

# Set EarlyStopping
EStop = EarlyStopping(monitor='val_loss', min_delta=0,patience=20, verbose=2,
                      mode='auto')

Best_model = ModelCheckpoint(filepath='weights.hdf5', monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max')

history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test),
                    epochs=150, batch_size=128, 
                    callbacks=[learning_rate_reduction, EStop, Best_model])

l = plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
print(l)
# Accuracy Curves
a = plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
print(a)
test_data = pd.read_csv('../input/test.csv')
test_data = test_data.fillna(0)
ids = test_data['Id']
Xt = test_data.drop(columns=['v18q1', 'rez_esc', 'idhogar', 'dependency', 'edjefe',
                           'edjefa',	'meaneduc', 'Id'],axis = 1)
Xt = Xt.astype('float32')
Xt = Xt.values
Xt = scaler.fit_transform(Xt)
pred = model.predict(Xt)
covertype = [np.argmax(i)+1 for i in pred]
sub = pd.DataFrame({'Id':ids,'Target':covertype})
output = sub[['Id','Target']]
output.to_csv("output.csv",index = False)