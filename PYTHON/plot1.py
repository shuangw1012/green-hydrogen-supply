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
plt.rcParams["font.family"] = "Times New Roman"

Loc = np.array(['KF249'])#,'KJ256','User'])

file_lowcost = '/output-KF249-Lined Rock.csv'
file_highcost = '/output-KF249-No_UG.csv'


df = pd.read_csv(os.getcwd()+ '/multiyear/' + file_lowcost)
X = np.linspace(1,8760*5,8760*5)

fig = plt.figure(figsize=(15, 5))
#plt.plot(X,df['pipe_storage_level'].values/1000)
plt.plot(X,df['ug_storage_level'].values/1000)
plt.ylabel('Storage level (tH$_2$)',fontsize = 14)
plt.xlabel('Hour (h)',fontsize = 14)
for i in range(5):
    plt.axvline(x=8760*(i+1), color='black', linestyle='--')
plt.ylim(-100,3000)
plt.xlim(0,8760*5)
plt.tight_layout()
plt.savefig(os.getcwd()+'/multiyear_Lined Rock.png',dpi=100)
plt.close(fig)

df = pd.read_csv(os.getcwd()+ '/multiyear/' + file_highcost)
X = np.linspace(1,8760*5,8760*5)

fig = plt.figure(figsize=(15, 5))
plt.plot(X,df['pipe_storage_level'].values/1000,color = 'orange')
plt.ylabel('Storage level (tH$_2$)',fontsize = 14)
plt.xlabel('Hour (h)',fontsize = 14)
for i in range(5):
    plt.axvline(x=8760*(i+1), color='black', linestyle='--')
plt.ylim(-100,1100)
plt.xlim(0,8760*5)
plt.tight_layout()
plt.savefig(os.getcwd()+'/multiyear_No_UG.png',dpi=100)
plt.close(fig)

df = pd.read_csv(os.getcwd()+ '/singleyear/' + file_lowcost)
X = np.linspace(1,8760,8760)
fig = plt.figure(figsize=(3, 5))
plt.plot(X,df['ug_storage_level'].values/1000)
plt.ylabel('Storage level (tH$_2$)',fontsize = 14)
plt.xlabel('Hour (h)',fontsize = 14)
plt.ylim(-100,3000)
plt.xlim(0,8760)
plt.tight_layout()
plt.savefig(os.getcwd()+'/singleyear_Lined Rock.png',dpi=100)
plt.close(fig)

df = pd.read_csv(os.getcwd()+ '/singleyear/' + file_highcost)
X = np.linspace(1,8760,8760)

fig = plt.figure(figsize=(3, 5))
plt.plot(X,df['pipe_storage_level'].values/1000,color = 'orange')
plt.ylabel('Storage level (tH$_2$)',fontsize = 14)
plt.xlabel('Hour (h)',fontsize = 14)
plt.ylim(-100,1100)
plt.xlim(0,8760)
plt.tight_layout()
plt.savefig(os.getcwd()+'/singleyear_No_UG.png',dpi=100)
plt.close(fig)