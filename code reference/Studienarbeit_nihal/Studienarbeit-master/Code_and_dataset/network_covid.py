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
    # target = df.pop('Cases_Per_Million')
    target = df[['Cases_Per_Million','west(0)/east(1)']]
    target = target.values
    features = df.drop(['Landkreis','Cases','LK_Einwohner','Deaths','Cases_Per_Million','Avg_Age','Std_dev','Density','west(0)/east(1)'],axis=1)
    features = features.values
    
    ix = np.random.choice(features.shape[0],int(features.shape[0]*0.2),replace=False)
    testfeatures = features[ix,:]
    testtarget=target[ix,:]
    features = np.delete(features,ix,axis=0)
    target=np.delete(target,ix,axis=0)



    def logCosh(y_true,y_pred):
        return tf.reduce_mean(keras.losses.logcosh(y_true,y_pred))

    model = keras.models.Sequential()
    model.add(keras.Input(shape=(5,)))
    model.add(keras.layers.Dense(50,activation=None))
    model.add(keras.layers.Dense(50,activation='elu'))
    model.add(keras.layers.Dense(50,activation='elu'))
    model.add(keras.layers.Dense(50,activation='elu'))
    model.add(keras.layers.Dense(50,activation='elu'))
    model.add(keras.layers.Dense(1))
    
    opt = keras.optimizers.Adam(lr=1e-2)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,0],epochs=25,validation_split=0.2)
    
    history = history.history
    fig, ax1 = plt.subplots()
    ax1.plot(history['loss'],label='Training Loss')
    ax1.plot(history['val_loss'],label='Val Loss')
    ax1.legend()
    fig.suptitle('Plot of Losses')

    target_predict = model.predict(x=testfeatures)
    
    # fig = plt.figure()
    # ax2 = fig.add_subplot(111, projection='3d')
    # cols = pltcolor(target[:,-1])    
    # ax2.scatter(features[:,0],df.values[:,-4],target[:,0],c=cols)
    # ax2.scatter(features[:,0],df.values[:,-4],target_predict)
    
    axises=['Income','Area','<30','30-65','>65']
    for i in range(0,2):
        for j in range (i+1,testfeatures.shape[1]):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            cols = pltcolor(testtarget[:,-1]) 
            ax.scatter(testfeatures[:,i],testfeatures[:,j],testtarget[:,0],c=cols)
            ax.scatter(testfeatures[:,i],testfeatures[:,j],target_predict)
            ax.set_xlabel(axises[i])
            ax.set_ylabel(axises[j])
            ax.set_zlabel('Cases Per million')
    
    # plt.scatter(np.linspace(1,target.size,target.size),target)
    # plt.scatter(np.linspace(1,target.size,target.size),target_predict)
    print(df.head())

if __name__ == "__main__":
    main()
