#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:41:16 2023

@author: admin-shuang
"""

import pandas as pd
import numpy as np
import os
from projdirs import datadir #load the path that contains the data files 
from PACKAGE.optimisation import Optimise
from PACKAGE.component_model import pv_gen, wind_gen, SolarResource, WindSource,WindSource_windlab
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
fontsize = 14
import time
from calendar import monthrange

def update_resource_data(Location):
    
    #Update the weather data files
    SolarResource(Location)
    
    # # WindSource(Location)
    WindSource_windlab(Location)

def wind_output(Location):
    from calendar import monthrange
    
    wind_ref = 320e3 #(kW)
    wind_ref_pout = list(np.trunc(100*np.array(wind_gen(Location)))/100)

    # Split the hourly data into monthly data
    monthly_output = []
    start_date = '2014-01-01'
    for year in range(2014, 2016):
        for month in range(1, 13):
            _, days_in_month = monthrange(year, month)
            monthly_output.append(sum(wind_ref_pout[:days_in_month * 24]))
            wind_ref_pout = wind_ref_pout[days_in_month * 24:]
    
    rated_capacity = wind_ref
    # Calculate the monthly capacity factor
    monthly_capacity_factor = [(output / (rated_capacity * (days_in_month * 24))) for output, days_in_month in zip(monthly_output, [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])]
    capacity_factors_arr = np.array(monthly_capacity_factor)[:12]
    #np.savetxt('%s/monthly_capacity_factor_%s.csv'%(os.getcwd(),Location), capacity_factors_arr, delimiter=',')
    return (np.average(capacity_factors_arr))

def solar_output(Location):
    
    pv_ref = 1e3 #(kW)
    pv_ref_pout = list(np.trunc(100*np.array(pv_gen(pv_ref)))/100)
    #print (sum(pv_ref_pout)/(pv_ref*8760))
    # Split the hourly data into monthly data
    monthly_output = []
    start_date = '2014-01-01'
    for year in range(2014, 2016):
        for month in range(1, 13):
            _, days_in_month = monthrange(year, month)
            monthly_output.append(sum(pv_ref_pout[:days_in_month * 24]))
            pv_ref_pout = pv_ref_pout[days_in_month * 24:]
    
    rated_capacity = pv_ref
    # Calculate the monthly capacity factor
    monthly_capacity_factor = [(output / (rated_capacity * (days_in_month * 24))) for output, days_in_month in zip(monthly_output, [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])]
    capacity_factors_arr = np.array(monthly_capacity_factor)[:12]
    #np.savetxt('%s/monthly_capacity_factor_%s.csv'%(os.getcwd(),Location), capacity_factors_arr, delimiter=',')
    return (np.average(capacity_factors_arr))

def CF_output(PV_location_g,Coor_PV_x_g,Coor_PV_y_g):
    
    CF = np.array([])
    for j in range(len(PV_location_g)):
        loc = PV_location_g[j]
        print 
        print (loc)
        update_resource_data(loc)
        solar_output(loc)
        CF = np.append(CF,[loc,Coor_PV_x_g[j],Coor_PV_y_g[j],round(solar_output(loc),3),round(wind_output(loc),3)])
    CF = CF.reshape(int(len(CF)/5),5)
    df = pd.DataFrame(CF, columns=['Location', 'Lat', 'Long', 'Solar CF', 'Wind CF'])
    df.to_csv(os.getcwd() + os.sep + 'CF_output.txt', sep=',', index=False, header=True)

def obtain_CC():
    weather_data_folder = datadir + os.sep + 'SAM_INPUTS' + os.sep + 'WEATHER_DATA'
    location1 = 'KI253'
    location2 = 'KF250'
    file1 = weather_data_folder + os.sep + 'weather_data_%s.csv'%location1
    file2 = weather_data_folder + os.sep + 'weather_data_%s.csv'%location2
    wspd1 = pd.read_csv(file1)['Pressure Units'].values[2:].astype(float)
    wspd2 = pd.read_csv(file2)['Pressure Units'].values[2:].astype(float)
    print (wspd1)
    corr_matrix = np.corrcoef(wspd1, wspd2)[0, 1]
    print (corr_matrix)

if __name__=='__main__':
    df = pd.read_csv(os.getcwd()+os.sep+'input_Northusg.txt')
    
    Wind_location = PV_location = df['#Name'].values
    Coor_wind_x = Coor_PV_x = df['Lat'].values
    Coor_wind_y = Coor_PV_y = df['Long'].values
    CF_output(PV_location,Coor_PV_x,Coor_PV_y)
    