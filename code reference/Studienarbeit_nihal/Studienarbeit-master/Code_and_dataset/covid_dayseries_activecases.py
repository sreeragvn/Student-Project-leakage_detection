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

lastday=129

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
    
df_cases = df_covid[df_covid['Landkreis'].isin(df1['Landkreis'])].reset_index(drop=True)
df_cases['firstday']=firstday
temp=pd.DataFrame(casedist)
temp['Landkreis']=df_cases['Landkreis']
df_cases=pd.merge(df_cases,temp)
df_cases=df_cases.drop(['Cases_Per_Million','Area','Avg_Age','Std_dev'],axis=1)
df_cases = pd.melt(df_cases, id_vars=['Landkreis','LK_Einwohner','Cases','Deaths','Density','Income','<30','30-65','>65','firstday','west(0)/east(1)'], 
              var_name="Day", value_name="DailyCases")
df_cases['Day'] = pd.to_numeric(df_cases['Day'],errors='coerce')
df_cases=df_cases.sort_values(['Landkreis','Day']).reset_index(drop=True)

################################################################################################

df1=df[['IdBundesland','Landkreis','AnzahlTodesfall','Meldedatum']]
df1=df1.groupby(['Landkreis','Meldedatum'],as_index=False)['AnzahlTodesfall'].sum()
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

case1 = df_covid1[df_covid1['Landkreis'].isin(df1['Landkreis'])] 
case2 = df1[df1['Landkreis'].isin(df_covid1['Landkreis'])]
df1 = pd.merge(case1,case2)

columns = list(df1) 
columns=columns[2:]
deathdist=np.zeros([df1['Landkreis'].size,lastday])

for index,row in df1.iterrows():
    cases=0
    count=0
    for i in columns: 
            cases=cases+row[i]
            deathdist[index,count]=cases
            count=count+1
     
df_death = df_covid[df_covid['Landkreis'].isin(df1['Landkreis'])].reset_index(drop=True)
temp=pd.DataFrame(deathdist)
temp['Landkreis']=df_death['Landkreis']
df_death=pd.merge(df_death,temp)
df_death=df_death.drop(['LK_Einwohner','Cases_Per_Million','Area','Avg_Age','Std_dev'],axis=1)
df_death = pd.melt(df_death, id_vars=['Landkreis','Cases','Deaths','Density','Income','<30','30-65','>65','west(0)/east(1)'], 
              var_name="Day", value_name="DailyDeaths")
df_death['Day'] = pd.to_numeric(df_death['Day'],errors='coerce')
df_death=df_death.sort_values(['Landkreis','Day']).reset_index(drop=True)
        
################################################################################################       
        
df1=df[['IdBundesland','Landkreis','AnzahlGenesen','Meldedatum']]
df1=df1.groupby(['Landkreis','Meldedatum'],as_index=False)['AnzahlGenesen'].sum()
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

case1 = df_covid1[df_covid1['Landkreis'].isin(df1['Landkreis'])] 
case2 = df1[df1['Landkreis'].isin(df_covid1['Landkreis'])]
df1 = pd.merge(case1,case2)

columns = list(df1) 
columns=columns[2:]
recoverdist=np.zeros([df1['Landkreis'].size,lastday+14])

for index,row in df1.iterrows():
    cases=0
    count=0
    for i in columns: 
            cases=cases+row[i]
            recoverdist[index,14+count]=cases
            count=count+1

recoverdist=recoverdist[:,:-14]     
df_rec = df_covid[df_covid['Landkreis'].isin(df1['Landkreis'])].reset_index(drop=True)
temp=pd.DataFrame(recoverdist)
temp['Landkreis']=df_rec['Landkreis']
df_rec=pd.merge(df_rec,temp)
df_rec=df_rec.drop(['LK_Einwohner','Cases_Per_Million','Area','Avg_Age','Std_dev'],axis=1)
df_rec = pd.melt(df_rec, id_vars=['Landkreis','Cases','Deaths','Density','Income','<30','30-65','>65','west(0)/east(1)'], 
              var_name="Day", value_name="DailyRecovery")
df_rec['Day'] = pd.to_numeric(df_rec['Day'],errors='coerce')
df_rec=df_rec.sort_values(['Landkreis','Day']).reset_index(drop=True)

################################################################################################   

df_active=df_cases.copy()
df_active['DailyDeaths']=df_death['DailyDeaths']
df_active['DailyRecovery']=df_rec['DailyRecovery']
df_active['ActiveCases']=df_active['DailyCases']-(df_active['DailyDeaths']+df_active['DailyRecovery'])
df_active.to_csv('DaySeriesActiveCovidDataSet.csv', header=True, index=False, encoding='utf-8')



