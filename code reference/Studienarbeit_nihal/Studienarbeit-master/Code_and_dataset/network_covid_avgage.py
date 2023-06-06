#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:20:40 2020

@author: nihal
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==0:
            cols.append('red')
        else:
            cols.append('orange')
    return cols


def main():
    df=pd.read_csv('CovidDataSet.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    # target = df.pop('Cases_Per_Million')
    target = df[['Deaths','west(0)/east(1)']]
    target = target.values
    features = df.drop(['Landkreis','Cases','Cases_Per_Million','Deaths','Density','<30','30-65','>65','west(0)/east(1)'],axis=1)
    features = features.values
    
    # ix = np.random.choice(features.shape[0],int(features.shape[0]*0.2),replace=False)
    # testfeatures = features[ix,:]
    # testtarget=target[ix,:]
    # features = np.delete(features,ix,axis=0)
    # target=np.delete(target,ix,axis=0)



    def logCosh(y_true,y_pred):
        return tf.reduce_mean(keras.losses.logcosh(y_true,y_pred))

    model = keras.models.Sequential()
    # model.add(keras.layers.Dropout(0.2,input_shape=(5,)))
    model.add(keras.layers.Dense(100,activation=None,input_dim=5,kernel_initializer='normal',kernel_constraint=max_norm(5)))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5))) 
  
    model.add(keras.layers.Dense(1))
    
    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,0:1],batch_size=8,epochs=25)#,validation_split=0.2)
    
    opt = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,0],batch_size=8,epochs=25)#,validation_split=0.2)

    opt = keras.optimizers.Adam(lr=1e-5)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,0:1],batch_size=8,epochs=25)#,validation_split=0.2)
    
    history = history.history
    fig, ax1 = plt.subplots()
    ax1.plot(history['loss'],label='Training Loss')
    # ax1.plot(history['val_loss'],label='Val Loss')
    ax1.legend()
    fig.suptitle('Plot of Losses')

    target_predict = model.predict(x=features)
    
    # fig = plt.figure()
    # ax2 = fig.add_subplot(111, projection='3d')
    # cols = pltcolor(target[:,-1])    
    # ax2.scatter(features[:,0],df.values[:,-4],target[:,0],c=cols)
    # ax2.scatter(features[:,0],df.values[:,-4],target_predict)
    
    # axises=['Income','Area','<30','30-65','>65']
    axises=['Income','LK_Einwohner','Area','Avg_Age','Std_dev']
    for i in range (0,features.shape[1]):
        for j in range (i+1,features.shape[1]):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            cols = pltcolor(target[:,-1]) 
            ax.scatter(features[:,i],features[:,j],target[:,0],c=cols)
            ax.scatter(features[:,i],features[:,j],target_predict[:,0],label='Predicted Data')
            ax.set_xlabel(axises[i])
            ax.set_ylabel(axises[j])
            ax.set_zlabel('Deaths')
            ax.legend()
    
    # plt.scatter(np.linspace(1,target.size,target.size),target)
    # plt.scatter(np.linspace(1,target.size,target.size),target_predict)
    print(df.head())

if __name__ == "__main__":
    main()
