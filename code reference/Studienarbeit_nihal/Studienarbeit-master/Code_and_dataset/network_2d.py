#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 22:58:59 2020

@author: nihal
"""


import tensorflow as tf
from tensorflow import keras
from funktion_2d import genData
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow.keras.backend as K

def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==0:
            cols.append('orange')
        else:
            cols.append('red')
    return cols
n=400
trainData,Z1_nonoise,Z2_nonoise  = genData(nSamples=n,frac=0.3)
X=trainData[:,0]
X=np.reshape(X,(-1,n)).T
Y=X.T
ix = np.random.choice(trainData.shape[0],int(trainData.shape[0]*0.05),replace=False)
testData = trainData[ix,:]
trainData = np.delete(trainData,ix,axis=0)

# fig = plt.figure()
# ax2 = fig.add_subplot(111, projection='3d')
# cols = pltcolor(testData[:,-1])
# ax2.scatter(X, Y, Z1_nonoise,color='orange',label= 'Function 1')
# ax2.scatter(X, Y, Z2_nonoise,color='red',label= 'Function 2')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# ax2.legend()
# fig.suptitle('Plot of 2D functions')

def logCosh(y_true,y_pred):
    return tf.reduce_mean(keras.losses.logcosh(y_true,y_pred))

model = keras.models.Sequential()
model.add(keras.layers.Dense(50,activation=None,input_dim=2))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(1,activation=None))

opt = keras.optimizers.Adam(lr=1e-4)
model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
# model.load_weights('weights')

val_split = 0.2
batch_size = 100
epochs = 10

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto',
                                                      )

hist = model.fit(x=trainData[:,:2],y=trainData[:,2],batch_size = batch_size, epochs = epochs)#,validation_split=val_split)
                  #callbacks=[early_stop])

predData = model.predict(testData[:,:2])

Xtest=testData[:,0]
Xtest=np.reshape(Xtest,(-1,n/5))
Ytest=testData[:,1]
Ytest=np.reshape(Ytest,(-1,n/5))
Zpred=predData
Zpred=np.reshape(Zpred,(-1,n/5))

history = hist.history
fig, ax1 = plt.subplots()
ax1.plot(history['loss'],label='Training Loss')
# ax1.plot(history['val_loss'],label='Val Loss')
ax1.legend()
fig.suptitle('Plot of Losses')

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
cols = pltcolor(testData[:,-1])
ax2.scatter(testData[:,0],testData[:,1],testData[:,2],c=cols,s=5,label='Test Data')
ax2.scatter(testData[:,0],testData[:,1],predData,s=6,label='Pred Data')
# ax2.plot_surface(Xtest, Ytest, Zpred,label='Pred Data')
ax2.plot_surface(X, Y, Z1_nonoise,color='orange')
ax2.plot_surface(X, Y, Z2_nonoise,color='red')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()
fig.suptitle('Test and Pred Data')