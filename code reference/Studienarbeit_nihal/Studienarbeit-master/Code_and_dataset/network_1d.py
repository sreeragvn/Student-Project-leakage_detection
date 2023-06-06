# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:52:04 2020

@author: Ajay N R
"""

import tensorflow as tf
from tensorflow import keras
from funktion_1d import genData
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow.keras.backend as K

def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==0:
            cols.append('orange')
        else:
            cols.append('blue')
    return cols
    
trainData,Z1_nonoise,Z2_nonoise = genData(nSamples=5000,frac=0.7)
X=trainData[:,0]
ix = np.random.choice(trainData.shape[0],int(trainData.shape[0]*0.1),replace=False)
testData = trainData[ix,:]
trainData = np.delete(trainData,ix,axis=0)

def logCosh(y_true,y_pred):
    return tf.reduce_mean(keras.losses.logcosh(y_true,y_pred))

model = keras.models.Sequential()
model.add(keras.layers.Dense(50,activation=None,input_shape=[1]))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(1,activation=None))

opt = keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
# model.load_weights('weights')

val_split = 0.2
batch_size = 100
epochs = 25

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto',
                                                     )

hist = model.fit(x=trainData[:,0],y=trainData[:,1],batch_size = batch_size, epochs = epochs)#,validation_split=val_split)
                 #callbacks=[early_stop])


predData = model.predict(testData[:,0])

history = hist.history
fig, ax1 = plt.subplots()
ax1.plot(history['loss'],label='Training Loss')
# ax1.plot(history['val_loss'],label='Val Loss')
ax1.legend()
fig.suptitle('Plot of Losses')

fig, ax2 = plt.subplots()
cols = pltcolor(testData[:,-1])
ax2.scatter(testData[:,0],testData[:,1],c=cols,label='Test Data')
ax2.scatter(testData[:,0],predData,label='Pred Data')
ax2.plot(X,Z1_nonoise,c='orange',linewidth='3')
ax2.plot(X,Z2_nonoise,c='blue',linewidth='3')
        
ax2.legend()
fig.suptitle('Test and Pred Data')

