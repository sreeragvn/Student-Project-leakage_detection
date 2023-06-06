#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:11:01 2020

@author: nihal
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import max_norm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
    
    lastday=129-7
    df=pd.read_csv('DaySeriesActiveCovidDataSetWithPast.csv')
    
    df = df.sample(frac=1).reset_index(drop=True)
    target = df[['ActiveCases','west(0)/east(1)']]
    target = target.values
    features = df[['Density','Income','firstday','0','1','2','3','4','5','6','Day']]
    features = features.values
    
    df=df.sort_values(['Landkreis','Day']).reset_index(drop=True)
    testfeatures = df[['Density','Income','firstday','0','1','2','3','4','5','6','Day']]
    testfeatures = testfeatures.values
    testtarget=df[['ActiveCases','west(0)/east(1)']]
    testtarget = testtarget.values
    
    model = keras.models.Sequential()
    # model.add(keras.layers.Dropout(0.2,input_shape=(6,)))
    model.add(keras.layers.Dense(50,activation=None,input_dim=11,kernel_initializer='normal',kernel_constraint=max_norm(5)))
    for i in range(0,15):
        model.add(keras.layers.Dense(50,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
        # model.add(keras.layers.BatchNormalization())
        # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))
    
    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,:1],batch_size=100 ,epochs=15)#,validation_split=0.1)
    
    opt = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,:1],batch_size=100 ,epochs=15)#,validation_split=0.1)
    
    opt = keras.optimizers.Adam(lr=1e-5)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,:1],batch_size=100 ,epochs=15)#,validation_split=0.1)   

    
    history = history.history
    fig, ax1 = plt.subplots()
    ax1.plot(history['loss'],label='Training Loss')
    # ax1.plot(history['val_loss'],label='Val Loss')
    ax1.legend()
    fig.suptitle('Plot of Losses')
    
    test_loss, test_acc = model.evaluate(features, target[:,:1])
    
    
    for i in range(0,349):
        target_predict = model.predict(x=testfeatures[i*lastday:(i+1)*lastday,:]) 
        
        fig, ax = plt.subplots()
        ax.plot(testfeatures[i*lastday:(i+1)*lastday,10],testtarget[i*lastday:(i+1)*lastday,0])
        ax.plot(testfeatures[i*lastday:(i+1)*lastday,10],target_predict)
        filename='plots_past/'+str(int(df['west(0)/east(1)'].iloc[i*lastday]))+df['Landkreis'].iloc[i*lastday]+'.png'
        ax.figure.savefig(filename)
        plt.close(fig)

    
if __name__ == "__main__":
    main()