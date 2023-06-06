#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 00:28:00 2020

@author: nihal
"""


from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt



def genData(n,noise):
    np.random.seed(0)
    
    f1 = lambda x: ((x-4)*(x+4))**2+np.random.normal(0,noise,x.size)
    f2 = lambda x: np.zeros(x.shape)+np.random.normal(0,noise,x.size)
    
    x1 = np.zeros((n*1/5,))
    y1 = np.zeros((n*1/5,))
    for i in range(n*1/5):
        x1[i] = np.sort(-6+ np.random.rand(1) *2)
        y1[i] = f1(x1[i])

    x2 = np.zeros((n*3/5,))
    y2 = np.zeros((n*3/5,))
    for i in range(n*3/5):
        x2[i] = np.sort(-4+ np.random.rand(1) *8)
        y2[i] = f2(x2[i])        

    x3 = np.zeros((n*1/5,))
    y3 = np.zeros((n*1/5,))
    for i in range(n*1/5):
        x3[i] = np.sort(4+ np.random.rand(1) *2)
        y3[i] = f1(x3[i])
         
    x = np.append(np.append(x1,x2),x3)  
    x = x.reshape((x.size,1))
    y = np.append(np.append(y1,y2),y3)    
    y = y.reshape((y.size,1))
    
    trainData = np.hstack((x,y))
    
    return trainData 


trainData = genData(2000,1)
ix = np.random.choice(trainData.shape[0],int(trainData.shape[0]*0.2),replace=False)
testData = trainData[ix,:]
trainData = np.delete(trainData,ix,axis=0)


model = keras.models.Sequential()
model.add(keras.layers.Dense(50,activation=None,input_dim=1))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(1))

opt = keras.optimizers.Adam(lr=1e-2)
model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])

# val_split = 0.2
batch_size = 32
epochs = 100

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto', )

hist = model.fit(x=trainData[:,0],y=trainData[:,1],batch_size = batch_size, epochs = epochs)#,validation_split=val_split)
                 
predData = model.predict(testData[:,0])

history = hist.history
fig, ax1 = plt.subplots()
ax1.plot(history['loss'],label='Training Loss')
# ax1.plot(history['val_loss'],label='Val Loss')
ax1.legend()
fig.suptitle('Plot of Losses')


fig, ax2 = plt.subplots()
ax2.scatter(testData[:,0],testData[:,1],label='Test Data')
ax2.scatter(testData[:,0],predData,label='Pred Data')
ax2.legend()
fig.suptitle('Test and Pred Data')
plt.show()

