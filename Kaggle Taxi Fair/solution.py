# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:05:22 2018

@author: david
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras import metrics
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('train.csv', nrows = 10000)
df = df.fillna(df.mean())
print(df.head())
Y = df['fare_amount'].values
X = df.drop(columns=['key', 'pickup_datetime', 'fare_amount'],axis = 1)
scaler = MinMaxScaler(feature_range = (0,1))
X = X.values
X = scaler.fit_transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(1))
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)

model.compile(optimizer ='nadam', loss = 'mean_squared_error', metrics =[metrics.mae])

history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test),
                    epochs=1500, batch_size=512, callbacks=[learning_rate_reduction])

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


df_t = pd.read_csv('test.csv')
ids = df_t['key']
Xt = df_t.drop(columns=['key', 'pickup_datetime'],axis = 1)
Xt = Xt.values
Xt = scaler.fit_transform(Xt)
pred = model.predict(Xt)
df_sub = pd.DataFrame({
    'key': df_t['key'].values,
    'fare_amount': pred[:,0]
}).set_index('key')
df_sub.head()
df_sub.to_csv('submission.csv')