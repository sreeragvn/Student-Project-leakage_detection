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
from numpy import inf
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
    
    lastday=129
    # df=pd.read_csv('DaySeriesCovidDataSetAgegroup.csv')
    df=pd.read_csv('DaySeriesCovidDataSet.csv')
    
    df_age=pd.read_csv('AgegroupCovidDataSet.csv')
    df_age = df_age[df_age['Landkreis'].isin(df['Landkreis'])].reset_index(drop=True)
    df['Einwohner(35-79)']=df_age['Einwohner(35-79)']
    
    df=df[df['Landkreis']!='Berlin']
    df=df[df['Landkreis']!='Hamburg']
    df=df[df['Landkreis']!='M\xc3\xbcnchen']
    
    # df = df.sample(frac=1).reset_index(drop=True)
    # target = df[['Total_Deaths','west(0)/east(1)']]
    df=df.drop(['Cases_Per_Million','Area','Avg_Age','Std_dev'],axis=1)
    df2 = pd.melt(df, id_vars=['Landkreis','LK_Einwohner','Cases','Deaths','Density','Income','<30','30-65','>65','Einwohner(35-79)','firstday','west(0)/east(1)'], 
                  var_name="Day", value_name="DailyCases")
    df2['Day'] = pd.to_numeric(df2['Day'],errors='coerce')
    df2=df2.sort_values(['Landkreis','Day']).reset_index(drop=True)
    df2['Relative(<30)']=df2['<30']/df2['LK_Einwohner']
    df2['Relative(30-65)']=df2['30-65']/df2['LK_Einwohner']
    df2['Relative(>65)']=df2['>65']/df2['LK_Einwohner']
    df2['RelativeEinwohner(35-79)']=df2['Einwohner(35-79)']/df2['LK_Einwohner']
    df2['LogDailycases']=np.log(df2['DailyCases'])
    df2['LogDailycases'][df2['LogDailycases']== -inf ]=0     
    
    df2['RelativeDailycases']=df2['DailyCases']/df2['LK_Einwohner']
    df2['RelativeLogDailycases']=np.log(df2['RelativeDailycases'])
    df2['RelativeLogDailycases'][df2['RelativeLogDailycases']== -inf ]= np.log(1.0/df2['LK_Einwohner'])       
    
    dfE = df2[df2['west(0)/east(1)'] == 1]
    dfE = dfE.sample(frac=1).reset_index(drop=True)
    target = dfE[['LogDailycases','west(0)/east(1)']]
    target = target.values
    # features = dfE[['Density','Income','Relative(<30)','Relative(30-65)','firstday','Day']]    
    features = dfE[['Density','Income','<30','30-65','>65','firstday','Day']]
    # features = dfE[['Density','Income','RelativeEinwohner(35-79)','firstday','Day']]
    features = features.values
    
    dfE = df2[df2['west(0)/east(1)'] == 1]
    # dfW=dfW.sort_values(['Landkreis','Day']).reset_index(drop=True)
    # # testfeatures = dfW[['Density','Income','Relative(<30)','Relative(30-65)','firstday','Day']]
    
    dfE=dfE.sort_values(['Landkreis','Day']).reset_index(drop=True)    
    testfeatures =  dfE[['Density','Income','<30','30-65','>65','firstday','Day']]
    # testfeatures =  dfW[['Density','Income','RelativeEinwohner(35-79)','firstday','Day']]
    testfeatures = testfeatures.values
    testtarget=dfE[['LogDailycases','west(0)/east(1)']]
    testtarget = testtarget.values


    
    model = keras.models.Sequential()
    # model.add(keras.layers.Dropout(0.2,input_shape=(6,)))
    model.add(keras.layers.Dense(50,activation=None,input_dim=7,kernel_initializer='normal',kernel_constraint=max_norm(5)))
    for i in range(0,15):
        model.add(keras.layers.Dense(50,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
        # model.add(keras.layers.BatchNormalization())
        # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))
    
    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,:1],batch_size=100 ,epochs=25)#,validation_split=0.1)
    
    # weights = model.get_weights()

    opt = keras.optimizers.Adam(lr=1e-4)
    # model.set_weights(weights)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,:1],batch_size=100 ,epochs=25)#,validation_split=0.1)
    
    opt = keras.optimizers.Adam(lr=1e-5)
    # model.set_weights(weights)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,:1],batch_size=100 ,epochs=25)#,validation_split=0.1)   
    
    # model_json = model.to_json()
    # with open("EastLog.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("EastLog.h5")

    # model_json = model.to_json()
    # with open("AgeGroupOmitBigCities/EastLogRelative.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("AgeGroupOmitBigCities/EastLogRelative.h5")
    
    history = history.history
    fig, ax1 = plt.subplots()
    ax1.plot(history['loss'],label='Training Loss')
    # ax1.plot(history['val_loss'],label='Val Loss')
    ax1.legend()
    fig.suptitle('Plot of Losses')
    
    test_loss, test_acc = model.evaluate(features, target[:,:1])
    
    testtarget[:,0]=np.exp(testtarget[:,0])
    
    for i in range(0,dfE.shape[0]/lastday):
        target_predict = model.predict(x=testfeatures[i*lastday:(i+1)*lastday,:]) 
        target_predict=np.exp(target_predict[:,0])  
        fig, ax = plt.subplots()
        ax.plot(testfeatures[i*lastday:(i+1)*lastday,6],testtarget[i*lastday:(i+1)*lastday,0],c='orange',label='Actual')
        ax.plot(testfeatures[i*lastday:(i+1)*lastday,6],target_predict,c='blue',label='Predicted')
        ax.set_xlabel("Age Of the pandemic (Start-28-01-2020)")
        # ax.set_xlabel("Age Of the pandemic (Start-02-02-2020)")
        ax.set_ylabel("Total Cases")
        ax.set_title("Plot of Total Cases")
        # ax.set_ylabel("Log of Total relative Cases")
        # ax.set_title("Log Plot of Total relative Cases")        
        ax.legend()
        filename='OmitBigCities/plotlogEastEastNormal/'+dfE['Landkreis'].iloc[i*lastday]+'.png'        
        # filename='plotlogEastWest/'+dfW['Landkreis'].iloc[i*lastday]+'.png'
        ax.figure.savefig(filename)


    
if __name__ == "__main__":
    main()