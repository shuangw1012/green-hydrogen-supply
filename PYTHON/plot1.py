#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:07:35 2023

@author: admin-shuang
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Loc = np.array(['KF249','KJ256','User'])
for loc in Loc:
    
    file = os.getcwd()+'/output-%s-Lined Rock.csv'%loc
    
    df = pd.read_csv(file)
    X = np.linspace(1,8760,8760)
    
    if loc == 'KF249':
        x1 = 5000
        x2 = 7000
    elif loc == 'KJ256':
        x1 = 800
        x2 = 2800
    elif loc == 'User':
        x1 = 1000
        x2 = 3000
    
    fig = plt.figure(figsize=(8, 5))
    plt.plot(X,df['ug_storage_level'].values/1000)
    plt.ylabel('Storage level (tH$_2$)',fontsize = 14)
    plt.xlabel('Hour (h)',fontsize = 14)
    plt.axvline(x=x1, color='black', linestyle='--')
    plt.axvline(x=x2, color='black', linestyle='--')
    plt.ylim(-100,4000)
    plt.xlim(0,8760)
    plt.tight_layout()
    #plt.tick_params(axis='both', labelsize=12)
    plt.savefig(os.getcwd()+'%s-storage.png'%loc,dpi=100)
    plt.close(fig)
    
    fig = plt.figure(figsize=(8, 5))
    plt.plot(X,df['wind_output'].values/1000,linewidth=0.2,color='red')
    plt.ylabel('Wind output (MWh)',fontsize = 14)
    plt.xlabel('Hour (h)',fontsize = 14)
    plt.axvline(x=x1, color='black', linestyle='--')
    plt.axvline(x=x2, color='black', linestyle='--')
    plt.ylim(-50,700)
    plt.xlim(0,8760)
    plt.tight_layout()
    #plt.tick_params(axis='both', labelsize=12)
    plt.savefig(os.getcwd()+'%s-wind.png'%loc,dpi=100)
    plt.close(fig)
        
    fig = plt.figure(figsize=(8, 5))
    X = np.linspace(x1,x2,x2-x1)
    plt.plot(X,df['wind_output'].values[x1:x2]/1000,linewidth=0.3,color='red')
    plt.ylabel('Wind output (MWh)',fontsize = 14)
    plt.xlabel('Hour (h)',fontsize = 14)
    plt.ylim(-50,700)
    plt.xlim(x1,x2)
    plt.tight_layout()
    #plt.tick_params(axis='both', labelsize=12)
    plt.savefig(os.getcwd()+'%s-wind-partial.png'%loc,dpi=100)
    plt.close(fig)