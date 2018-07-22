# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 08:58:30 2018

@author: david
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('train.csv')
df['Sex'].replace('female', 1,inplace=True)
df['Sex'].replace('male', 0,inplace=True)
print(df.head())
Y = df['Survived'].values
X = df.drop(columns=['Name', 'PassengerId', 'Ticket', 'Cabin', 
                     'Embarked', 'Survived'],axis = 1)
Y = Y[:418]
X = X[:418]
X = X.fillna(X.mean())
scaler = MinMaxScaler(feature_range = (0,1))
X = X.values

X = scaler.fit_transform(X)
Y = pd.get_dummies(Y)
Y = Y.values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,
                                                 random_state=42)

model = Sequential()
model.add(Dense(30, input_dim=X_train.shape[1], activation='elu'))
model.add(Dense(25, activation ='elu'))
model.add(Dense(25, activation ='elu'))
model.add(Dense(25, activation ='elu'))
model.add(Dense(25, activation ='elu'))
model.add(Dense(2, activation = "sigmoid"))

model.compile(optimizer = 'adam' , loss = "binary_crossentropy", 
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
                    epochs=50, batch_size=128, 
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
dt = pd.read_csv('test.csv')
ids = dt['PassengerId']
dt['Sex'].replace('female', 1,inplace=True)
dt['Sex'].replace('male', 0,inplace=True)
Xt = dt.drop(columns=['Name', 'PassengerId', 'Ticket', 'Cabin', 
                     'Embarked'],axis = 1)
Xt = Xt.fillna(Xt.mean())
scaler = MinMaxScaler(feature_range = (0,1))
Xt = Xt.values
Xt = scaler.fit_transform(X)
predictions = model.predict_classes(Xt, verbose=1)
submissions=pd.DataFrame({"PassengerId": dt["PassengerId"],"Survived": predictions})
submissions.to_csv("survived.csv", index=False, header=True)