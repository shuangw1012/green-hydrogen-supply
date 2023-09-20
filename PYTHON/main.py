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

def update_resource_data(Location):
    #Choose the location
    #Location = 'Gladstone 3_MERRA2' 
    
    #Update the weather data files
    SolarResource(Location)
    
    # # WindSource(Location)
    WindSource_windlab(Location)
    
def optimisation():
    # create a dictionary that contains the inputs for optimisation.
    #these inputs are used by make_dzn_file function to create an input text file called hydrogen_plant_data.dzn.                 
    
    # Pipe storage costs
    # line packing: 516 USD/kgH2
    # Ardent UG storage: 110-340 USD/kgH2
    # vessel storage: 1000 USD/kgH2
    
    
    #for 2020
    simparams = dict(EL_ETA = 0.70,       #efficiency of electrolyser
                     BAT_ETA_in = 0.95,   #charging efficiency of battery
                     BAT_ETA_out = 0.95,  #discharg efficiency of battery
                     C_PV = 1122.7,          #[USD/kW] unit cost of PV
                     C_WIND = 1455,           #[USD/kW] unit cost of Wind
                     C_EL = 1067,          #[USD/W] unit cost of electrolyser
                     UG_STORAGE_CAPA_MAX = 1e10,   #maximum available salt caevern size (kg of H2)
                     C_PIPE_STORAGE = 516, #unit cost of line packing (USD/kg of H2)
                     PIPE_STORAGE_CAPA_MIN = 0, #minimum size of linepacking (kg of H2)
                     C_BAT_ENERGY = 196,        #[USD/kWh] unit cost of battery energy storage
                     C_BAT_POWER = 405,        #[USD/kW] unit cost of battery power capacpity
                     )
    
    # # for 2030
    # simparams = dict(EL_ETA = 0.70,       #efficiency of electrolyser
    #                  BAT_ETA_in = 0.95,   #charging efficiency of battery
    #                  BAT_ETA_out = 0.95,  #discharg efficiency of battery
    #                  C_PV = 696,          #[USD/kW] unit cost of PV
    #                  C_WIND = 1390,           #[USD/kW] unit cost of Wind
    #                  C_EL = 385,          #[USD/W] unit cost of electrolyser
    #                  UG_STORAGE_CAPA_MAX = 0,   #maximum available salt caevern size (kg of H2)
    #                  C_PIPE_STORAGE = 1000, #unit cost of line packing (USD/kg of H2)
    #                  PIPE_STORAGE_CAPA_MIN = 0, #minimum size of linepacking (kg of H2)
    #                  C_BAT_ENERGY = 164,        #[USD/kWh] unit cost of battery energy storage
    #                  C_BAT_POWER = 338,        #[USD/kW] unit cost of battery power capacpity
    #                  ) 
    
    # # for 2050
    # simparams = dict(EL_ETA = 0.70,       #efficiency of electrolyser
    #                  BAT_ETA_in = 0.95,   #charging efficiency of battery
    #                  BAT_ETA_out = 0.95,  #discharg efficiency of battery
    #                  C_PV = 465,          #[USD/kW] unit cost of PV
    #                  C_WIND = 1323,           #[USD/kW] unit cost of Wind
    #                  C_EL = 295,          #[USD/W] unit cost of electrolyser
    #                  UG_STORAGE_CAPA_MAX = 0,   #maximum available salt caevern size (kg of H2)
    #                  C_PIPE_STORAGE = 1000, #unit cost of line packing (USD/kg of H2)
    #                  PIPE_STORAGE_CAPA_MIN = 0, #minimum size of linepacking (kg of H2)
    #                  C_BAT_ENERGY = 131,        #[USD/kWh] unit cost of battery energy storage
    #                  C_BAT_POWER = 270,        #[USD/kW] unit cost of battery power capacpity
    #                  ) 
    
    
    #Choose the location
    
    # #Update the weather data files
    # SolarResource(Location)
    
    # # # WindSource(Location)
    # WindSource_windlab(Location)
    
    # storage_type = 'Lined Rock'
    # results = Optimise(5, 100, storage_type, simparams)
    
    
    import multiprocessing as mp
    
    # for Location in ['Pilbara 2', 'Pilbara 3', 'Pilbara 4', 'Burnie 1', 'Burnie 2', 'Burnie 3', 'Burnie 4',
    #                'Pinjara 1', 'Pinjara 2', 'Pinjara 3', 'Pinjara 4',
    #                'Upper Spencer Gulf 1', 'Upper Spencer Gulf 2', 'Upper Spencer Gulf 3', 'Upper Spencer Gulf 4',
    #                'Gladstone 1', 'Gladstone 2', 'Gladstone 3']:
        
    #for Location in ['Burnie 1','Burnie 2','Burnie 3','Burnie 4']:

        #Update the weather data files
        #SolarResource(Location)
    
        # # WindSource(Location)
        #WindSource_windlab(Location)
        
    #Location = ['Burnie 1','Burnie 2']#,'Burnie 3','Burnie 4']
    #output = Optimise(load=5, cf=80, storage_type='Salt Cavern', simparams=simparams,Location=Location)
    '''
    pool = mp.Pool(mp.cpu_count()-2)
    print('Started!')
    output = [pool.apply_async(Optimise,
                               args=(load, CF, storage_type, params))
              for load in [2.115]
              for CF in [80]#[50,60,70,80,90,100]
              for storage_type in ['Salt Cavern'] 
              for params in [simparams]]

    pool.close()
    pool.join()
    print('Completed!')
    '''
    CF_group = [50,60]
    output = []
    for i in range(2):
        CF = CF_group[i]
        feedback = Optimise(load=5, cf=CF, storage_type='Salt Cavern', simparams=simparams)
        print (feedback)
        output.append(feedback)
    
    RESULTS = pd.DataFrame(columns=['cf','capex[USD]','pv_capacity[kW]',
                                    'wind_capacity[kW]','pv_capacity_array[kW]',
                                    'wind_capacity_array[kW]','el_capacity[kW]',
                                    'ug_capcaity[kgH2]','pipe_storage_capacity[kgH2]',
                                    'bat_e_capacity[kWh]','bat_p_capacity[kW]',
                                    'pv_cost[USD]', 'wind_cost[USD]','el_cost[USD]',
                                    'ug_storage_cost[USD]','pipe_storage_cost[USD]',
                                    'bat_cost[USD]', 'load[kg/s]','C_trans[USD]'])
    
    for i in range(len(output)):
        results = output[i]
        RESULTS = RESULTS.append({'cf': results['CF'],
                            'capex[USD]': results['CAPEX'][0],
                            'pv_capacity[kW]': results['pv_max'][0],
                            'wind_capacity[kW]': results['wind_max'][0],
                            'pv_capacity_array[kW]': results['pv_max_array'],
                            'wind_capacity_array[kW]': results['wind_max_array'],
                            'el_capacity[kW]': results['el_max'][0],
                            'ug_capcaity[kgH2]': results['ug_storage_capa'][0],
                            'pipe_storage_capacity[kgH2]': results['pipe_storage_capa'][0],
                            'bat_e_capacity[kWh]': results['bat_e_capa'][0],
                            'bat_p_capacity[kW]': results['bat_p_max'][0],
                            'pv_cost[USD]': results['pv_max'][0]*simparams['C_PV'],
                            'wind_cost[USD]': results['wind_max'][0]*simparams['C_WIND'],
                            'el_cost[USD]': results['el_max'][0]*simparams['C_EL'],
                            'ug_storage_cost[USD]': results['ug_storage_capa'][0]*results['C_UG_STORAGE'],
                            'pipe_storage_cost[USD]':results['pipe_storage_capa'][0]*simparams['C_PIPE_STORAGE'],
                            'bat_cost[USD]': results['bat_p_max'][0]*simparams['C_BAT_ENERGY'],
                            'load[kg/s]':results['LOAD'][0],
                            'C_trans[USD]':results['C_trans'][0]}, ignore_index=True)


    #RESULTS
    parent_directory = os.path.dirname(os.getcwd())
    path_to_file = parent_directory + os.sep + 'DATA' + os.sep + 'OPT_OUTPUTS' + os.sep 
    result_file = 'results_2020.csv'

    RESULTS.to_csv(path_to_file+result_file, index=False)


def wind_output(Location):
    from calendar import monthrange
    
    wind_ref = 320e3 #(kW)
    wind_ref_pout = list(np.trunc(100*np.array(wind_gen()))/100)

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
    np.savetxt('%s/monthly_capacity_factor_%s.csv'%(os.getcwd(),Location), capacity_factors_arr, delimiter=',')

def plot(location):

    # Example data (replace these arrays with your actual data)
    array1 = np.genfromtxt('%s/monthly_capacity_factor_%s.csv'%(os.getcwd(),location[0]), delimiter=',')
    array2 = np.genfromtxt('%s/monthly_capacity_factor_%s.csv'%(os.getcwd(),location[1]), delimiter=',')
    array3 = np.genfromtxt('%s/monthly_capacity_factor_%s.csv'%(os.getcwd(),location[2]), delimiter=',')
    array4 = np.genfromtxt('%s/monthly_capacity_factor_%s.csv'%(os.getcwd(),location[3]), delimiter=',')
    
    # Replace 'months1' and 'capacity_factors1' with your first set of data
    # Replace 'months2' and 'capacity_factors2' with your second set of data
    months1 = np.arange(1, 13)-0.225  # Assuming 24 months from 1 to 24
    capacity_factors1 = array1  # Capacity factors for 24 months
    
    months2 = np.arange(1, 13)-0.075  # Assuming 12 months from 1 to 12
    capacity_factors2 = array2 # Capacity factors for 12 months
    
    months3 = np.arange(1, 13)+0.075  # Assuming 12 months from 1 to 12
    capacity_factors3 = array3 # Capacity factors for 12 months
    
    months4 = np.arange(1, 13)+0.225  # Assuming 12 months from 1 to 12
    capacity_factors4 = array4 # Capacity factors for 12 months
    
    plt.figure(figsize=(10, 8))
    plt.bar(months1, capacity_factors1, width=0.15, color='b', edgecolor='black', label='Wlab')
    plt.bar(months2, capacity_factors2, width=0.15, color='green', edgecolor='black', label='BARRA')
    plt.bar(months3, capacity_factors3, width=0.15, color='r', edgecolor='black', label='MERRA2')
    plt.bar(months4, capacity_factors4, width=0.15, color='black', edgecolor='black', label='ERA5')
    
    # Add labels and legend
    months = range(1, 13)
    plt.xticks(months, fontsize=fontsize)
    plt.xlabel('Month',fontsize=fontsize)
    plt.ylabel('Capacity Factor',fontsize=fontsize)
    #plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.ylim(0,0.7)
    plt.savefig('%s/comparison_%s.png'%(os.getcwd(),location[0]), dpi=500, bbox_inches='tight')

def plot_yearly():
    title = np.array(['Burnie 1', 'Burnie 2', 'Burnie 3', 'Burnie 4', 'Gladstone 1', 'Gladstone 2', 'Gladstone 3', 
                      'Pilbara 1', 'Pilbara 2', 'Pilbara 3', 'Pilbara 4', 'Pinjarra 1', 'Pinjarra 2', 'Pinjarra 3', 'Pinjarra 4',
                      'USG 1', 'USG 2', 'USG 3', 'USG 4'])
    Data = np.array([[0.484,0.422,0.580,0.530,0.505],
                     [0.572,0.551,0.645,0.568,0.65],
                     [0.487,0.491,0.567,0.522,0.63],
                     [0.369,0.416,0.571,0.459,0.415],
                     [0.356,0.206,0.411,0.389,0.35],
                     [0.375,0.207,0.371,0.410,0.2],
                     [0.375,0.397,0.432,0.453,0.38],
                     [0.426,0.352,0.438,0.480,0.46],
                     [0.374,0.237,0.294,0.447,0.38],
                     [0.423,0.421,0.415,0.468,0.48],
                     [0.426,0.400,0.417,0.453,0.48],
                     [0.499,0.366,0.437,0.519,0.49],
                     [0.499, 0.428, 0.395, 0.489, 0.38],
                     [0.495, 0.405, 0.465, 0.523, 0.55],
                     [0.494, 0.405, 0.428, 0.502, 0.46],
                     [0.419, 0.362, 0.482, 0.478, 0.45],
                     [0.449, 0.358, 0.475, 0.526, 0.47],
                     [0.435, 0.388, 0.499, 0.523, 0.5],
                     [0.415, 0.362, 0.428, 0.478, 0.405]])
    
    Index = np.linspace(1,len(title),len(title))
    plt.figure(figsize=(16, 8))
    plt.bar(Index-0.3, Data[:,3], width=0.15, color='b', edgecolor='black', label='Wlab')
    plt.bar(Index-0.15, Data[:,2], width=0.15, color='green', edgecolor='black', label='BARRA')
    plt.bar(Index, Data[:,0], width=0.15, color='r', edgecolor='black', label='MERRA2')
    plt.bar(Index+0.15, Data[:,1], width=0.15, color='black', edgecolor='black', label='ERA5')
    plt.bar(Index+0.3, Data[:,4], width=0.15, color='pink', edgecolor='black', label='Atlas')
    
    #plt.xlabel(title,fontsize=fontsize,rotation=45)
    plt.ylabel('Capacity Factor',fontsize=fontsize)
    plt.xticks(Index-0.5,title,fontsize=fontsize,rotation=45)
    plt.yticks(fontsize=fontsize)
    plt.legend(ncol=5,fontsize=fontsize)
    plt.ylim(0,0.7)
    plt.savefig('%s/comparison_yearly.png'%(os.getcwd()), dpi=500, bbox_inches='tight')

def solar_output(Location):
    from calendar import monthrange
    
    pv_ref = 1e3 #(kW)
    pv_ref_pout = list(np.trunc(100*np.array(pv_gen(pv_ref)))/100)

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
    np.savetxt('%s/monthly_capacity_factor_%s.csv'%(os.getcwd(),Location), capacity_factors_arr, delimiter=',')

if __name__=='__main__':
    #optimisation()
    '''
    Loc = ['Burnie 1', 'Burnie 2', 'Burnie 3', 'Burnie 4']
    
    for j in range(len(Loc)):
        loc = Loc[j]
        print 
        print (loc)
        location = loc+'_MERRA2'
        update_resource_data(loc)
        solar_output(loc)
    '''
    #plot(location)
    #plot_yearly()
    optimisation()