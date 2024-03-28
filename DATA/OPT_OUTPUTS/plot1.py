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


file_1 = '/output-DN48-Salt Cavern.csv'
file_2 = '/output-Pilbara 1-Salt Cavern.csv'

df = pd.read_csv(os.getcwd() + file_1)
df2 = pd.read_csv(os.getcwd() + file_2)

X = np.linspace(1,8760,8760)

fig = plt.figure(figsize=(8, 5))
#plt.plot(X,df['pipe_storage_level'].values/1000)
plt.plot(X,df['ug_storage_level'].values/1000,label = 'DN48')
plt.plot(X,df2['ug_storage_level'].values/1000,color='r',label='Pilbara 1')
plt.ylabel('Storage level (tH$_2$)',fontsize = 14)
plt.xlabel('Hour (h)',fontsize = 14)
#for i in range(5):
#    plt.axvline(x=8760*(i+1), color='black', linestyle='--')
plt.ylim(-100,4000)
plt.xlim(0,8760)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.getcwd()+'/storage.png',dpi=100)
plt.close(fig)
