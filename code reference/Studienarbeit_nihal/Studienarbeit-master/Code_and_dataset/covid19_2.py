#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:38:54 2020

@author: nihal
"""


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    df_covid = pd.read_csv(r'RKI_COVID19.csv')  
    df_covid=df_covid[['IdBundesland','Landkreis','Altersgruppe','AnzahlFall','AnzahlTodesfall']]
    
    df_cases=df_covid.groupby(['Landkreis','Altersgruppe'],as_index=False)['AnzahlFall'].sum()
    df_cases=pd.pivot_table(df_cases, index='Landkreis', columns='Altersgruppe', aggfunc=np.sum, fill_value=0).reset_index()
    df_cases.columns = df_cases.columns.droplevel()
    df_cases.rename(columns={list(df_cases)[0]:'Landkreis'}, inplace=True)
    df_cases['0-34']=df_cases['A00-A04']+df_cases['A05-A14']+df_cases['A15-A34']
    df_cases['35-79']=df_cases['A35-A59']+df_cases['A60-A79']
    df_cases['80+']=df_cases['A80+']
    df_cases=df_cases[['Landkreis','0-34','35-79','80+']].copy()
    df_cases['Total_Cases']=df_cases['0-34']+df_cases['35-79']+df_cases['80+']
    df_cases['Landkreis']=df_cases['Landkreis'].str[3:]
    for i in range(0,df_cases['Landkreis'].size):
        if 'Berlin' in df_cases['Landkreis'].iloc[i] :
            df_cases['Landkreis'].iloc[i]='Berlin'
        elif 'Aachen'in df_cases['Landkreis'].iloc[i] :
            df_cases['Landkreis'].iloc[i]='Aachen'        
    df_cases = pd.pivot_table(df_cases, index=['Landkreis'],values=['0-34','35-79','80+','Total_Cases'],aggfunc='sum').reset_index()
    df_cases['Landkreis'] = [unicode(item,"utf-8") for item in df_cases['Landkreis']]
    
    df_deaths=df_covid.groupby(['Landkreis','Altersgruppe'],as_index=False)['AnzahlTodesfall'].sum()
    df_deaths=pd.pivot_table(df_deaths, index='Landkreis', columns='Altersgruppe', aggfunc=np.sum, fill_value=0).reset_index()
    df_deaths.columns = df_deaths.columns.droplevel()
    df_deaths.rename(columns={list(df_deaths)[0]:'Landkreis'}, inplace=True)
    df_deaths['D(0-34)']=df_deaths['A00-A04']+df_deaths['A05-A14']+df_deaths['A15-A34']
    df_deaths['D(35-79)']=df_deaths['A35-A59']+df_deaths['A60-A79']
    df_deaths['D(80+)']=df_deaths['A80+']
    df_deaths=df_deaths[['Landkreis','D(0-34)','D(35-79)','D(80+)']].copy()
    df_deaths['Total_Deaths']=df_deaths['D(0-34)']+df_deaths['D(35-79)']+df_deaths['D(80+)']
    df_deaths['Landkreis']=df_deaths['Landkreis'].str[3:]
    for i in range(0,df_deaths['Landkreis'].size):
        if 'Berlin' in df_deaths['Landkreis'].iloc[i] :
            df_deaths['Landkreis'].iloc[i]='Berlin'
        elif 'Aachen'in df_deaths['Landkreis'].iloc[i] :
            df_deaths['Landkreis'].iloc[i]='Aachen'        
    df_deaths = pd.pivot_table(df_deaths, index=['Landkreis'],values=['D(0-34)','D(35-79)','D(80+)','Total_Deaths'],aggfunc='sum').reset_index()
    df_deaths['Landkreis'] = [unicode(item,"utf-8") for item in df_deaths['Landkreis']]
    
    case1 = df_cases[df_cases['Landkreis'].isin(df_deaths['Landkreis'])] 
    case2 = df_deaths[df_deaths['Landkreis'].isin(df_cases['Landkreis'])]
    df = pd.merge(case1,case2)
    
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
    df_income['Landkreis'] = [unicode(item,"utf-8") for item in df_income['Landkreis']]
    
    case1 = df_income[df_income['Landkreis'].isin(df['Landkreis'])] 
    case2 = df[df['Landkreis'].isin(df_income['Landkreis'])]
    df = pd.merge(case1,case2)
    
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
    
    df['Density'] = df['Einwohner']/df['Area']
    
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
    
    df=df[df['Landkreis'].isin(df_agegroup['Landkreis'])].reset_index()
    df['Einwohner(0-34)']=df_agegroup[['<3','3-6','6-10','10-15','15-18','18-20','20-25','25-30','30-35']].sum(axis=1)
    df['Einwohner(35-79)']=df_agegroup[['35-40','40-45','45-50','50-55','55-60','60-65','65-75']].sum(axis=1)
    df['Einwohner(80+)']=df_agegroup[['>75']].sum(axis=1)

    df_dup = df_ref[df_ref.duplicated('Landkreis',keep='first')]
    df_ref.drop_duplicates(['Landkreis'],inplace=True)
    df_ref=df_ref[df_ref['Landkreis'].isin(df['Landkreis'])].reset_index()
    df_ref=df_ref.sort_values(by=['Ref_no'])
    
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
    
    df.pop('index')
    df.to_csv('AgegroupCovidDataSet.csv', header=True, index=False, encoding='utf-8')
            
    cols = df.columns.drop(['Landkreis','west(0)/east(1)'])
    dfnew=df[cols]
    corr = dfnew.corr()
    
    # plot the heatmap
    sns_plot = sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
    fig = sns_plot.get_figure()
    fig.savefig("./AgegroupHeatMap.png")


if __name__ == "__main__":
    main()