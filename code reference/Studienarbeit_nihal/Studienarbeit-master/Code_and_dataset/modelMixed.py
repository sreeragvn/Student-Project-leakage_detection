#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:17:26 2020

@author: nihal
"""
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def genData(n,noise):
    np.random.seed(0)
    
    f1 = lambda x: ((x-4)*(x+4))**2+np.random.normal(0,noise,x.size)
    f2 = lambda x: np.zeros(x.shape)+np.random.normal(0,noise,x.size)
    
    x2_1 = np.zeros((n*1/5,))
    y2_1 = np.zeros((n*1/5,))
    for i in range(n*1/5):
        x2_1[i] = np.sort(-6+ np.random.rand(1) *2)
        y2_1[i] = f1(x2_1[i])

    x2_2 = np.zeros((n*3/5,))
    y2_2 = np.zeros((n*3/5,))
    for i in range(n*3/5):
        x2_2[i] = np.sort(-4+ np.random.rand(1) *8)
        y2_2[i] = f2(x2_2[i])        

    x2_3 = np.zeros((n*1/5,))
    y2_3 = np.zeros((n*1/5,))
    for i in range(n*1/5):
        x2_3[i] = np.sort(4+ np.random.rand(1) *2)
        y2_3[i] = f1(x2_3[i])
         
    x = np.append(np.append(x2_1,x2_2),x2_3)  

    y2 = np.append(np.append(y2_1,y2_2),y2_3)    
    y2 = y2.reshape((y2.size,1))
    
    y1 = np.zeros((n,))
    y1=f1(x)
    y1 = y1.reshape((y1.size,1))
    x = x.reshape((x.size,1))    
    
    return x,y1,y2 

x,y1,y2=genData(4000,2)
ix1 = np.random.choice(x.size, int(np.round(0.5*x.size)), replace=False)
trainData = np.hstack([x[ix1],y1[ix1]])
ix2 = [y for y in range(x.size) if not y in ix1]
trainData = np.vstack([trainData,np.hstack([x[ix2],y2[ix2]])])
trainData = trainData[trainData[:,0].argsort()]  

# fig, ax= plt.subplots()
# ax.scatter(trainData[:,0],trainData[:,1])
# ax.scatter(x,y1)
# ax.scatter(x,y1-y2)
# ax.scatter(x,y2)
ix = np.random.choice(trainData.shape[0],int(trainData.shape[0]*0.2),replace=False)
testData = trainData[ix,:]
trainData = np.delete(trainData,ix,axis=0)


model = keras.models.Sequential()
model.add(keras.layers.Dense(50,activation=None,input_dim=1))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(1))

opt = keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])


batch_size = 32
epochs = 25

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto', )

hist = model.fit(x=trainData[:,0],y=trainData[:,1],batch_size = batch_size, epochs = epochs)              
predData = model.predict(testData[:,0])

history = hist.history
fig, ax1 = plt.subplots()
ax1.plot(history['loss'],label='Training Loss')
ax1.legend()
fig.suptitle('Plot of Losses')


fig, ax2 = plt.subplots()
ax2.scatter(testData[:,0],testData[:,1],label='Test Data')
ax2.scatter(testData[:,0],predData,label='Pred Data')
ax2.legend()
fig.suptitle('Test and Pred Data')
plt.show()





