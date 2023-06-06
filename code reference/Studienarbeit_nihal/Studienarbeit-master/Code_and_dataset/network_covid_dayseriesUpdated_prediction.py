#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:45:22 2020

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

def main():
    
    lastday=125 #125/129
    df=pd.read_csv('DaySeriesCovidDataSetAgegroup.csv')
    # df=pd.read_csv('DaySeriesCovidDataSet.csv')

    df=df.drop(['Cases_Per_Million','Area','Avg_Age','Std_dev'],axis=1)
    
    df_age=pd.read_csv('AgegroupCovidDataSet.csv')
    df_age = df_age[df_age['Landkreis'].isin(df['Landkreis'])].reset_index(drop=True)
    df['Einwohner(35-79)']=df_age['Einwohner(35-79)']        
    
    df=df[df['Landkreis']!='Berlin']
    df=df[df['Landkreis']!='Hamburg']
    df=df[df['Landkreis']!='M\xc3\xbcnchen']
    
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
    
    df2 = df2.sample(frac=1).reset_index(drop=True)
    # target = df2[['RelativeLogDailycases','west(0)/east(1)']]
    target = df2[['LogDailycases','west(0)/east(1)']]
    target = target.values
    Einwohner=df2['LK_Einwohner']
    Einwohner=Einwohner.values
    # features = df2[['Density','Income','Relative(<30)','Relative(30-65)','firstday','Day']]
    # features = df2[['Density','Income','<30','30-65','>65','firstday','Day']]
    features = df2[['Density','Income','Einwohner(35-79)','firstday','Day']]
    features = features.values
    
    df2=df2.sort_values(['Landkreis','Day']).reset_index(drop=True)
    # testfeatures = df2[['Density','Income','Relative(<30)','Relative(30-65)','firstday','Day']]    
    # testfeatures = df2[['Density','Income','<30','30-65','>65','firstday','Day']]
    testfeatures = df2[['Density','Income','Einwohner(35-79)','firstday','Day']]
    testfeatures = testfeatures.values
    testEinwohner=df2['LK_Einwohner']
    testEinwohner=testEinwohner.values
    # testtarget = df2[['RelativeLogDailycases','west(0)/east(1)']]
    testtarget=df2[['LogDailycases','west(0)/east(1)']]
    testtarget = testtarget.values
    
    json_file = open('AgeGroupOmitBigCities/Totallog.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("AgeGroupOmitBigCities/TotalLog.h5")
    
    # testtarget[:,0]=np.exp(testtarget[:,0])*testEinwohner
    testtarget[:,0]=np.exp(testtarget[:,0])
    
    for i in range(0,df.shape[0]):
        target_predict = model.predict(x=testfeatures[i*lastday:(i+1)*lastday,:]) 
        
        # target_predict=np.exp(target_predict[:,0])*testEinwohner[i*lastday:(i+1)*lastday]
        target_predict=np.exp(target_predict[:,0])        
                
        fig, ax = plt.subplots()
        ax.plot(testfeatures[i*lastday:(i+1)*lastday,4],testtarget[i*lastday:(i+1)*lastday,0],c='orange',label='Actual')
        ax.plot(testfeatures[i*lastday:(i+1)*lastday,4],target_predict,c='blue',label='Predicted')
        # ax.set_xlabel("Age Of the pandamic (Start-28-01-2020)")
        ax.set_xlabel("Age Of the pandamic (Start-02-02-2020)")
        ax.set_ylabel("Total Cases")
        ax.set_title("Plot of Total Cases")
        # ax.set_ylabel("Log of Total relative Cases")
        # ax.set_title("Log Plot of Total relative Cases")
        ax.legend()           
        filename='AgeGroupOmitBigCities/plotsLogNormal/'+str(int(df['west(0)/east(1)'].iloc[i]))+df['Landkreis'].iloc[i]+'.png'
        # filename='plotsLog/'+str(int(df['west(0)/east(1)'].iloc[i]))+df['Landkreis'].iloc[i]+'.png'        
        ax.figure.savefig(filename)    
    
    
    
    
if __name__ == "__main__":
    main()    