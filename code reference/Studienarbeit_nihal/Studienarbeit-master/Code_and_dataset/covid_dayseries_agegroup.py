#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:45:47 2020

@author: nihal
"""


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df= pd.read_csv(r'RKI_COVID19.csv')  
df_covid = pd.read_csv(r'CovidDataSet.csv')

lastday=125

df=df[(df['Altersgruppe']=='A35-A59') | (df['Altersgruppe']=='A60-A79')]

df1=df[['IdBundesland','Landkreis','AnzahlFall','Meldedatum']]
df1=df1.groupby(['Landkreis','Meldedatum'],as_index=False)['AnzahlFall'].sum()
df1=pd.pivot_table(df1, index='Landkreis', columns=['Meldedatum'], aggfunc=np.sum, fill_value=0).reset_index()
df1.columns = df1.columns.droplevel()
df1.rename(columns={list(df1)[0]:'Landkreis'}, inplace=True)

df1['Landkreis']=df1['Landkreis'].str[3:]
for i in range(0,df1['Landkreis'].size):
    if 'Berlin' in df1['Landkreis'].iloc[i] :
        df1['Landkreis'].iloc[i]='Berlin'
    elif 'Aachen'in df1['Landkreis'].iloc[i] :
        df1['Landkreis'].iloc[i]='Aachen'        
        
df1 = pd.pivot_table(df1, index=['Landkreis'],aggfunc='sum').reset_index()

df_covid1=df_covid[['Landkreis','LK_Einwohner']].copy()
case1 = df_covid1[df_covid1['Landkreis'].isin(df1['Landkreis'])] 
case2 = df1[df1['Landkreis'].isin(df_covid1['Landkreis'])]
df1 = pd.merge(case1,case2)


columns = list(df1) 
columns=columns[2:]
casedist=np.zeros([df1['Landkreis'].size,lastday])

for index,row in df1.iterrows():
    cases=0
    count=0
    for i in columns: 
            cases=cases+row[i]
            casedist[index,count]=cases
            count=count+1

firstday=np.zeros([df1['Landkreis'].size])
firstcase=np.zeros([df1['Landkreis'].size])
for index,row in df1.iterrows():
    cases=0
    for i in columns: 
        if (cases/row['LK_Einwohner'])<0.00001:
            cases=cases+row[i]     
        else :
            firstday[index]=df1.columns.get_loc(i)-2
            firstcase[index]=cases   
            break            
       
df_covid = df_covid[df_covid['Landkreis'].isin(df1['Landkreis'])].reset_index(drop=True)
df_covid['firstday']=firstday
temp=pd.DataFrame(casedist)
temp['Landkreis']=df_covid['Landkreis']
df_covid=pd.merge(df_covid,temp)
df_covid.to_csv('DaySeriesCovidDataSetAgegroup.csv', header=True, index=False, encoding='utf-8')


        
        
        

