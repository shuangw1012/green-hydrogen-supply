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
import random

def optimisation(df_slice):
    for num in range(len(df_slice)):
        df_input = df_slice.iloc[num]
        
        simparams = dict(EL_ETA = 0.70,       #efficiency of electrolyser
                         BAT_ETA_in = 0.95,   #charging efficiency of battery
                         BAT_ETA_out = 0.95,  #discharg efficiency of battery
                         C_PV = df_input['PV_capex'],  #[USD/kW] unit cost of PV
                         C_WIND = df_input['Wind_capex'],#1894,#           #[USD/kW] unit cost of Wind
                         C_EL = df_input['El_capex'],          #[USD/W] unit cost of electrolyser
                         UG_STORAGE_CAPA_MAX = 1e10,   #maximum available salt caevern size (kg of H2)
                         C_PIPE_STORAGE = 516, #unit cost of line packing (USD/kg of H2)
                         PIPE_STORAGE_CAPA_MIN = 0, #minimum size of linepacking (kg of H2)
                         C_BAT_ENERGY = 196*df_input['Bat_ratio'],        #[USD/kWh] unit cost of battery energy storage
                         C_BAT_POWER = 405*df_input['Bat_ratio'],        #[USD/kW] unit cost of battery power capacpity
                         OM_EL = df_input['EL_FOM'],    # O&M for electrolyzer ($/kw)
                         OM_PV = df_input['PV_FOM'],    # O&M for PV ($/kw)
                         OM_WIND = df_input['Wind_FOM'],    # O&M for wind ($/kw)
                         OM_UG = 1.03*df_input['Stg_ratio'],        # O&M for underground storage ($/kg)
                         DIS_RATE = 0.06,        #discount rate 6%
                         C_stg_ratio = df_input['Stg_ratio'], # ratio of storage cost
                         C_trans_ratio = df_input['Trans_capex'], # ratio of transmission capex
                         C_pipe_ratio = df_input['Pipe_capex'], # ratio of pipeline capex
                         storage_type = df_input['Stg']
                         )
        
        load = df_input['load'] #0.2115, 0.705, 2.115, 7.0501, 21.1506
        year = int(df_input['data_year'])
        CF = df_input['CF']
        df = pd.read_csv(os.getcwd()+os.sep+'Input_Tas.txt')
        random_number = random.random()
        
        #import multiprocessing as mp
        
        output = []
        Simparams = []
        
        # we set Wind locations the same as PV for now
        wind_location = pv_location = df['#Name'].values[:-2]
        coor_wind_x = coor_PV_x = df['Lat'].values[:-2]
        coor_wind_y = coor_PV_y = df['Long'].values[:-2]
        el_location = df['#Name'].values[:-1]
        coor_el_x = df['Lat'].values[:-1]
        coor_el_y = df['Long'].values[:-1]
        #Pipe_buffer = df[df['Within_buffer']==True]['#Name'].values
        Area_list = df['Area'].values[:-2].tolist()
        #print ('Started CF: %s Case: %s'%(CF,pv_location))
        
        # transmission cost unit capacity
        km_per_degree = 111.32 # km/deg
        detour = 1.2 # detour factor
        
        # transmission cost
        coor_PV_x = coor_PV_x[:, np.newaxis]
        coor_PV_y = coor_PV_y[:, np.newaxis] 
        coor_wind_x = coor_wind_x[:, np.newaxis]
        coor_wind_y = coor_wind_y[:, np.newaxis]
        
        distancePV = np.sqrt((coor_PV_x - coor_el_x)**2 + (coor_PV_y - coor_el_y)**2)*km_per_degree*detour
        distanceWind = np.sqrt((coor_wind_x - coor_el_x)**2 + (coor_wind_y - coor_el_y)**2)*km_per_degree*detour
        
        # pipe cost
        user_x = df[df['#Name']=='User']['Lat'].values[0]
        user_y = df[df['#Name']=='User']['Long'].values[0]
        storage_x = df[df['#Name']=='storage']['Lat'].values[0]
        storage_y = df[df['#Name']=='storage']['Long'].values[0]
        
        distanceUser = np.sqrt((user_x - coor_el_x)**2 + (user_y - coor_el_y)**2)*km_per_degree*detour
        
        if storage_x!=0:
            distanceStg = np.sqrt((storage_x - coor_el_x)**2 + (storage_y - coor_el_y)**2)*km_per_degree*detour
        else:
            distanceStg = np.sqrt((storage_x - coor_el_x)**2 + (storage_y - coor_el_y)**2)*km_per_degree*detour*0
        
        pv_location = pv_location + '_%s'%year
        wind_location = wind_location + '_%s'%year
        feedback,simparams = Optimise(load, CF, simparams,pv_location,wind_location,
                                      Area_list,distancePV,distanceWind,distanceUser,distanceStg,random_number)
        
        output.append(feedback)
        Simparams.append(simparams)
        
        data_list = []
        parent_directory = os.path.dirname(os.getcwd())
        path_to_file = parent_directory + os.sep + 'DATA' + os.sep + 'OPT_OUTPUTS' + os.sep 
        
        for i in range(len(output)):
            results = output[i]
            simparams = Simparams[i]
            
            row_data = {
                'Hub': df_input['Hub'],'Stg': df_input['Stg'],'data_year': df_input['data_year'],
                'CF': df_input['CF'],'cost_year': df_input['cost_year'],'PV_capex': df_input['PV_capex'],
                'Wind_capex': df_input['Wind_capex'],'El_capex': df_input['El_capex'],'UG_capex': df_input['UG_capex'],
                'PV_FOM': df_input['PV_FOM'],'Wind_FOM': df_input['Wind_FOM'],
                'EL_FOM': df_input['EL_FOM'],'Stg_ratio': df_input['Stg_ratio'],
                'Bat_ratio': df_input['Bat_ratio'],
                'Trans_capex': df_input['Trans_capex'],'Pipe_capex': df_input['Pipe_capex'],
                'El': el_location[int(results['El_location'][0])-1], # Minizinc starts from 1
                'capex[USD]': results['CAPEX'][0],'lcoh[USD/kg]': results['lcoh'][0],
                'FOM_PV[USD]':results['FOM_PV'][0],'FOM_WIND[USD]':results['FOM_WIND'][0],
                'FOM_EL[USD]':results['FOM_EL'][0],'FOM_UG[USD]':results['FOM_UG'][0],
                'FOM_TRANS[USD]':results['FOM_TRANS'][0],'FOM_PIPE[USD]':results['FOM_PIPE'][0],
                'H_total[kg]':results['H_total'][0],'pv_capacity[kW]': results['pv_max'][0],
                'wind_capacity[kW]': results['wind_max'][0],'el_capacity[kW]': results['el_max'][0],
                'ug_capcaity[kgH2]': results['ug_storage_capa'][0],'pipe_storage_capacity[kgH2]': results['pipe_storage_capa'][0],
                'bat_e_capacity[kWh]': results['bat_e_capa'][0],'bat_p_capacity[kW]': results['bat_p_max'][0],
                'pv_cost[USD]': results['pv_max'][0] * simparams['C_PV'], 'wind_cost[USD]': results['wind_max'][0] * simparams['C_WIND'],
                'el_cost[USD]': results['el_max'][0] * simparams['C_EL'],'ug_storage_cost[USD]': results['ug_storage_capa'][0] * simparams['C_UG_STORAGE'],
                'pipe_storage_cost[USD]': results['pipe_storage_capa'][0] * simparams['C_PIPE_STORAGE'],
                'bat_cost[USD]': results['bat_p_max'][0] * simparams['C_BAT_ENERGY'],
                'load[kg/s]': results['LOAD'][0],'C_trans[USD]': results['C_trans'][0],
                'C_pipe[USD]': results['C_pipe'][0]
            }
            n_project = 25
            DIS_RATE = 0.06
            crf = DIS_RATE * (1+DIS_RATE)**n_project/((1+DIS_RATE)**n_project-1)
            H_total = results['H_total'][0]
            row_data['LCOH-PV']=(crf*results['pv_max'][0] * simparams['C_PV']+results['FOM_PV'][0])/H_total
            row_data['LCOH-wind']=(crf*results['wind_max'][0] * simparams['C_WIND']+results['FOM_WIND'][0])/H_total
            row_data['LCOH-el']=(crf*results['el_max'][0] * simparams['C_EL']+results['FOM_EL'][0])/H_total
            row_data['LCOH-UG']=(crf*results['ug_storage_capa'][0] * simparams['C_UG_STORAGE']+results['FOM_UG'][0])/H_total
            row_data['LCOH-pipe-storage']=(crf*results['pipe_storage_capa'][0] * simparams['C_PIPE_STORAGE'])/H_total
            row_data['LCOH-trans']=(crf*results['C_trans'][0]+results['FOM_TRANS'][0])/H_total
            row_data['LCOH-pipe']=(crf*results['C_pipe'][0]+results['FOM_PIPE'][0])/H_total
            
            if len(results['pv_max_array'])>1:
                for j in range(len(results['pv_max_array'])):
                    row_data['pv_capacity_%s[kW]'%pv_location[j]] = results['pv_max_array'][j]
                for j in range(len(results['wind_max_array'])):
                    row_data['wind_capacity_%s[kW]'%wind_location[j]] = results['wind_max_array'][j]
            data_list.append(row_data)
            
            output_series = True
            if output_series == True:
                df = pd.DataFrame({'pipe_storage_level': results['pipe_storage_level'][:-1],
                                   'ug_storage_level': results['ug_storage_level'][:-1],
                                   'pv_pout': results['pv_pout'],
                                   'wind_output': results['wind_pout'],
                                   'curtail_p':results['curtail_p'],
                                   'bat_pin':results['bat_pin'],
                                   'bat_pout':results['bat_pout'],
                                   'el_pin_pvwind':results['el_pin_pvwind'],
                                   'el_pin':results['el_pin'],
                                   'comp1_pin':results['comp1_pin'],
                                   'comp2_pin':results['comp2_pin'],
                                   'pipe_storage_hout': results['pipe_storage_hout'],
                                   'ug_storage_hout':results['ug_storage_hout'],
                                   'comp1_hflow':results['comp1_hflow'],
                                   'comp2_hflow':results['comp2_hflow'],
                                   'res_hout':results['res_hout'],
                                   'LOAD':results['LOAD']})
                
                df.to_csv(path_to_file+'output_2020_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.csv'%(
                    df_input['Hub'],df_input['Stg'],df_input['load'],
                    df_input['data_year'],df_input['CF'],df_input['cost_year'],df_input['PV_capex'],df_input['Wind_capex'],df_input['El_capex'],df_input['UG_capex'],df_input['PV_FOM'],
                    df_input['Wind_FOM'],df_input['EL_FOM'],df_input['Stg_ratio'],df_input['Bat_ratio'],df_input['Trans_capex'],df_input['Pipe_capex']), index=False)
            
            # clean up template weather data
            dir = datadir + os.sep + 'SAM_INPUTS' + os.sep + 'SOLAR' + os.sep 
            if os.path.exists(dir + 'SolarSource_%s.csv'%random_number):
                os.remove(dir + 'SolarSource_%s.csv'%random_number)
                
            dir = datadir + os.sep + 'SAM_INPUTS' + os.sep + 'WIND' + os.sep 
            if os.path.exists(dir + 'WindSource_%s.srw'%random_number):
                os.remove(dir + 'WindSource_%s.srw'%random_number)
            
        # Convert list of dictionaries to DataFrame
        RESULTS = pd.DataFrame(data_list)
        RESULTS.to_csv(path_to_file+'results_2020_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.csv'%(
            df_input['Hub'],df_input['Stg'],df_input['load'],
            df_input['data_year'],df_input['CF'],df_input['cost_year'],df_input['PV_capex'],df_input['Wind_capex'],df_input['El_capex'],df_input['UG_capex'],df_input['PV_FOM'],
            df_input['Wind_FOM'],df_input['EL_FOM'],df_input['Stg_ratio'],df_input['Bat_ratio'],df_input['Trans_capex'],df_input['Pipe_capex']))
        
if __name__=='__main__':
    
    df = pd.read_csv(os.getcwd()+os.sep+'input.txt')
    num_opt = 1
    #i=0
    #optimisation(df.iloc[i*num_opt:(i+1)*num_opt])
    
    from mpi4py import MPI
    for i in range(len(df)):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == i:
            #try:
            optimisation(df.iloc[i*num_opt:(i+1)*num_opt])
            #except:
            #    print ('Failed %s'%i)
    