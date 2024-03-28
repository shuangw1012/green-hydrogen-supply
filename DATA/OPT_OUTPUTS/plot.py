#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:20:05 2024

@author: admin-shuang
"""
import matplotlib.pyplot as plt
import numpy as np

import os

plt.rcParams["font.family"] = "Times New Roman"
import pandas as pd

file_name = os.getcwd()+'/weather_data_DN48.csv'
df = pd.read_csv(file_name)
wind = df['Pressure Units'].values[2:]

for i in range(8760):
    wind[i] = float(wind[i])

H = np.linspace(1,8760,8760)

fig = plt.figure(figsize=(8, 5))
plt.plot(H, wind,linewidth=0.5)
plt.xlim(0,8760)
plt.ylabel('Wind speed (m/s)',fontsize = 14)
plt.xlabel('Hours (h)',fontsize = 14)
plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig(os.getcwd()+'/wspd.png',dpi=100)
plt.close(fig)

Input = np.array([ 0,3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10, 10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75, 14, 14.25, 14.5, 14.75, 15, 15.25, 15.5, 15.75, 16, 16.25, 16.5, 16.75, 17, 17.25, 17.5, 17.75, 18, 18.25, 18.5, 18.75, 19, 19.25, 19.5, 19.75, 20, 20.25, 20.5, 20.75, 21, 21.25, 21.5, 21.75, 22, 22.25, 22.5, 22.75, 23, 23.25, 23.5, 23.75, 24, 24.25, 24.5, 24.75, 25, 30 ])
Output = np.array([0,0, 94, 132, 175, 225, 281, 344, 415, 494, 581, 676, 781, 895, 1019, 1153, 1298, 1455, 1623, 1802, 1995, 2200, 2418, 2650, 2896, 3156, 3431, 3722, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 0, 0 ])

fig = plt.figure(figsize=(8, 5))
plt.plot(Input, Output,linewidth=1,color='orange')
#plt.xlim(0,8760)
plt.ylabel('Power output(MW)',fontsize = 14)
plt.xlabel('Wind speed (m/s)',fontsize = 14)
plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig(os.getcwd()+'/wspd2.png',dpi=100)
plt.close(fig)
