#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:19:38 2020

@author: nihal
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    df_income = pd.read_csv(r'incomelandkreis.csv')  
    df_income['Income'] = pd.to_numeric(df_income['Income'],errors='coerce')
    df_income['Income']=df_income['Income']*1000
    df_income['Landkreis']=df_income['Landkreis'].str.split(',',1).str[0]
    df_income['Landkreis']=df_income['Landkreis'].str.rstrip(' ')
    df_income['Einwohner'] = df_income['Einwohner'].str.replace('.','')
    df_income['Einwohner'] = pd.to_numeric(df_income['Einwohner'],errors='coerce')
    df_income['Total Income'] = df_income['Einwohner']*df_income['Income']
    df_income = df_income.drop(['Income'],axis=1)
    df_dup = df_income[df_income.duplicated('Landkreis',keep='first')]
    df_income.drop_duplicates(['Landkreis'],inplace=True)
    df_income = df_income.set_index('Landkreis').add(df_dup.set_index('Landkreis'),fill_value=0).reset_index()
    df_income['Income'] = df_income['Total Income']/df_income['Einwohner']
    df_income = df_income.drop(['Total Income'],axis=1)
    df_income = df_income.drop(['Einwohner'],axis=1)
    df_income['Landkreis'] = [unicode(item,"utf-8") for item in df_income['Landkreis']]
    
    df_covid = pd.read_json(r'covid.json')
    columns = ['Landkreis','LK_Einwohner','Cases','Cases_Per_Million','Deaths']
    df_covid = df_covid[columns]
    df_covid=df_covid.sort_values(by=['Landkreis'])
    df_covid=df_covid.reset_index(drop=True)
    
    for i in range(0,df_covid['Landkreis'].size):
        if 'Berlin' in df_covid['Landkreis'].iloc[i] :
            df_covid['Landkreis'].iloc[i]='Berlin'
        elif df_covid['Landkreis'].iloc[i].count('(')==1 :
            df_covid['Landkreis'].iloc[i]=df_covid['Landkreis'].iloc[i].split(' (', 1)[0]
        else:
            df_covid['Landkreis'].iloc[i]=" (".join(df_covid['Landkreis'].iloc[i].split(" (", 2)[:2])   
            
        # df_covid['Landkreis'].iloc[i]=df_covid['Landkreis'].iloc[i].rstrip(' ')
    df_covid = df_covid.drop(['Cases_Per_Million'],axis=1)
    df_dup = df_covid[df_covid.duplicated('Landkreis',keep='first')]
    df_covid.drop_duplicates(['Landkreis'],inplace=True)
    df_dup = pd.pivot_table(df_dup, index=['Landkreis'],values=['LK_Einwohner','Cases','Deaths'],aggfunc='sum').reset_index()
    df_covid = df_covid.set_index('Landkreis').add(df_dup.set_index('Landkreis'),fill_value=0).reset_index()
    df_covid['Cases_Per_Million'] = df_covid['Cases']/df_covid['LK_Einwohner']*1000000    
    
    
    case1 = df_income[df_income['Landkreis'].isin(df_covid['Landkreis'])] 
    case2 = df_covid[df_covid['Landkreis'].isin(df_income['Landkreis'])]
    df = pd.merge(case1,case2)
    
    
    # df_income = pd.read_csv(r'income2.csv')
    # df_income['Net_needs'] = pd.to_numeric(df_income['Net_needs'],errors='coerce')
    # df_income['Gross_needs'] = pd.to_numeric(df_income['Gross_needs'],errors='coerce')
    # df_income['Ded_income'] = pd.to_numeric(df_income['Ded_income'],errors='coerce')
    # df_income=df_income.sort_values(by=['Landkreis'])
    # df_income=df_income.dropna()
    # df_income=df_income.reset_index(drop=True)  
    
    # df_income['Landkreis']=df_income['Landkreis'].str.split(',',1).str[0]
    # # df_income['Landkreis']=df_income['Landkreis'].map(lambda x: x.split(',',1)[0])
    # # print(df_income.head())
   
    # case1 = df_covid[df_covid['Landkreis'].isin(df_income['Landkreis'])] 
    # case2 = df_income[df_income['Landkreis'].isin(df_covid['Landkreis'])]
    # df = pd.merge(case1,case2)
    # print(df)
    
    
    # fig, ax = plt.subplots()
 
    # ax.scatter(df['Net_needs'],df['Cases_Per_Mill89ion'])
                
    df_area = pd.read_csv(r'Area.csv')  
    df_area['Ref_no']=pd.to_numeric(df_area['Ref_no'],errors='coerce')
    df_area['Area']=pd.to_numeric(df_area['Area'],errors='coerce')
    df_area=df_area.dropna()    
    df_area['Landkreis']=df_area['Landkreis'].str.split(',',1).str[0]
    
    df_ref=df_area[['Landkreis','Ref_no']].copy()
    df_ref['Landkreis'] = [unicode(item,"utf-8") for item in df_ref['Landkreis']]
    
    df_area=df_area.drop(['Ref_no'],axis=1)
    df_dup = df_area[df_area.duplicated('Landkreis',keep='first')]
    df_area.drop_duplicates(['Landkreis'],inplace=True)
    df_area = df_area.set_index('Landkreis').add(df_dup.set_index('Landkreis'),fill_value=0).reset_index()
    df_area['Landkreis'] = [unicode(item,"utf-8") for item in df_area['Landkreis']]

    case1 = df[df['Landkreis'].isin(df_area['Landkreis'])] 
    case2 = df_area[df_area['Landkreis'].isin(df['Landkreis'])]
    df = pd.merge(case1,case2)    

    df_agegroup = pd.read_csv(r'Agegroup.csv') 
    cols = df_agegroup.columns.drop('Landkreis')
    df_agegroup[cols] = df_agegroup[cols].apply(pd.to_numeric, errors='coerce')
    df_agegroup=df_agegroup.dropna()
    df_agegroup['Landkreis']=df_agegroup['Landkreis'].str.split(',',1).str[0]
    df_dup = df_agegroup[df_agegroup.duplicated('Landkreis',keep='first')]
    df_agegroup.drop_duplicates(['Landkreis'],inplace=True)
    df_agegroup = df_agegroup.set_index('Landkreis').add(df_dup.set_index('Landkreis'),fill_value=0).reset_index()
    df_agegroup['Landkreis'] = [unicode(item,"utf-8") for item in df_agegroup['Landkreis']]
    df_agegroup=df_agegroup[df_agegroup['Landkreis'].isin(df['Landkreis'])].reset_index()
    
    cols=cols[0:-1]
    avg=np.zeros(df_agegroup['Landkreis'].size)
    std=np.zeros(df_agegroup['Landkreis'].size)
    mid=np.array([1.5,4.5,8,12.5,16.5,19,22.5,27.5,32.5,37.5,42.5,47.5,52.5,57.5,62.5,70,87])
    for i in range(0,df_agegroup['Landkreis'].size):
        avg[i]=np.sum(df_agegroup[cols].iloc[i]*mid)/df_agegroup['Total'].iloc[i]
        std[i]=np.sqrt(np.sum(df_agegroup[cols].iloc[i]*np.square(mid))/df_agegroup['Total'].iloc[i]-np.square(avg[i]))  
    
    
    df_age=pd.DataFrame({'Landkreis' : df_agegroup['Landkreis'],
                        'Avg_Age' : avg,
                        'Std_dev' : std})     
      
    case1 = df[df['Landkreis'].isin(df_age['Landkreis'])] 
    case2 = df_age[df_age['Landkreis'].isin(df['Landkreis'])]
    df = pd.merge(case1,case2)    
    
    df_dup = df_ref[df_ref.duplicated('Landkreis',keep='first')]
    df_ref.drop_duplicates(['Landkreis'],inplace=True)
    df_ref=df_ref[df_ref['Landkreis'].isin(df['Landkreis'])] 
    df_ref=df_ref.sort_values(by=['Ref_no'])
                
    df['<30']=df_agegroup[['<3','3-6','6-10','10-15','15-18','18-20','20-25','25-30']].sum(axis=1)
    df['30-65']=df_agegroup[['30-35','35-40','40-45','45-50','50-55','55-60','60-65']].sum(axis=1)
    df['>65']=df_agegroup[['65-75','>75']].sum(axis=1)
    df['Density'] = df['LK_Einwohner']/df['Area']
    
    east=[]
    west=[]
    for i in range(0,df_ref['Landkreis'].size):
        if int(df_ref['Ref_no'].iloc[i]/1000)<=11 :
            west.append(df_ref['Landkreis'].iloc[i])
        else:
            east.append(df_ref['Landkreis'].iloc[i])
    
    df['west(0)/east(1)']=np.zeros(df['Landkreis'].size)
    for i in range(0,df['Landkreis'].size):
        if df['Landkreis'].iloc[i] in east:
            df['west(0)/east(1)'].iloc[i]=1
    
    df.to_csv('CovidDataSet.csv', header=True, index=False, encoding='utf-8')
    df_ref.to_csv('Landkreis.csv', header=True, index=False, encoding='utf-8')
    
    cols = df.columns.drop('Landkreis')
    dfnew=df[cols]
    corr = dfnew.corr()

# plot the heatmap
    sns_plot = sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
    fig = sns_plot.get_figure()
    fig.savefig("./HeatMap.png")

if __name__ == "__main__":
    main()
