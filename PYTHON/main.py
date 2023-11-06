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
                     OM_EL = 37.40,    # O&M for electrolyzer ($/kw)
                     OM_PV = 12.70,    # O&M for PV ($/kw)
                     OM_WIND = 18.65,    # O&M for wind ($/kw)
                     OM_UG = 1.03,        # O&M for underground storage ($/kg)
                     DIS_RATE = 0.06        #discount rate 6%
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
    
    PV_location_g,Coor_PV_x_g,Coor_PV_y_g,El_location_g,Coor_elx_x_g,Coor_elx_y_g,user_x,user_y,Pipe_buffer,Area = load_txt()
    
    # #Update the weather data files
    # SolarResource(Location)
    
    import multiprocessing as mp
    
    CF_group = [100]
    output = []
    Simparams = []
    
    # update resource data
    #Resource_data(PV_location_g,Coor_PV_x_g,Coor_PV_y_g)
    
    # we set Wind locations the same as PV for now
    Wind_location_g = PV_location_g
    Coor_wind_x_g = Coor_PV_x_g
    Coor_wind_y_g = Coor_PV_y_g
    
    
    # get the locations within pipe buffer
    Pipe_buffer_g = np.array([])
    for i in range(len(PV_location_g)):
        if Pipe_buffer[i] == 'True':
            Pipe_buffer_g = np.append(Pipe_buffer_g,PV_location_g[i])
    
    for CF in CF_group:        
        # adding a loop for different El locations
        for e in range(len(Coor_elx_x_g)):
            Coor_elx = Coor_elx_x_g[e]
            Coor_ely = Coor_elx_y_g[e]
            El_location = El_location_g[e]
            
            for j in range(len(PV_location_g)+1):
                if j < len(PV_location_g):
                    if PV_location_g[j] != El_location:
                        continue
                    PV_location = [PV_location_g[j]]
                    Wind_location = [Wind_location_g[j]]
                    Coor_PV_x = [Coor_PV_x_g[j]]
                    Coor_PV_y = [Coor_PV_y_g[j]]
                    Coor_wind_x = [Coor_wind_x_g[j]]
                    Coor_wind_y = [Coor_wind_y_g[j]]
                    Area_list = [1e6] # assume unlimited capacity if one location chosen
                if j == len(PV_location_g):
                    continue
                    PV_location = PV_location_g
                    Wind_location = Wind_location_g
                    Coor_PV_x = Coor_PV_x_g
                    Coor_PV_y = Coor_PV_y_g
                    Coor_wind_x = Coor_wind_x_g
                    Coor_wind_y = Coor_wind_y_g
                    Area_list = Area.tolist()
                    
                print ('Started CF: %s Case: %s %s'%(CF,PV_location,El_location_g[e]))
                
                # transmission cost unit capacity
                km_per_degree = 111.32 # km/deg
                C_PV_t = np.zeros(len(PV_location))
                C_wind_t = np.zeros(len(Wind_location))
                for i in range(len(PV_location)):
                    C_PV_t[i] = np.sqrt(abs((Coor_PV_x[i]-Coor_elx)**2+(Coor_PV_y[i]-Coor_ely)**2))*km_per_degree*5.496*0.67
                for i in range(len(Wind_location)):
                    C_wind_t[i] = np.sqrt(abs((Coor_wind_x[i]-Coor_elx)**2+(Coor_wind_y[i]-Coor_ely)**2))*km_per_degree*5.496*0.67
                C_PV_t = C_PV_t.tolist()
                C_wind_t = C_wind_t.tolist()
                      
                # pipe cost
                C_pipe = np.sqrt(abs((user_x-Coor_elx)**2+(user_y-Coor_ely)**2))*km_per_degree*589346.11*0.67
                if El_location in Pipe_buffer_g:
                    C_pipe = C_pipe*0.15 # USD
                feedback,simparams = Optimise(2.115, CF, 'Lined Rock', simparams,PV_location,Wind_location,
                                              C_PV_t,C_wind_t,C_pipe,Area_list)
                
                feedback['El']=El_location_g[e] # add el location to the results
                output.append(feedback)
                Simparams.append(simparams)
    
    data_list = []

    for i in range(len(output)):
        results = output[i]
        simparams = Simparams[i]
        
        row_data = {
            'cf': simparams['CF'],
            'El': results['El'],
            'capex[USD]': results['CAPEX'][0],
            'lcoh[USD/kg]': results['lcoh'][0],
            'FOM_PV[USD]':results['FOM_PV'][0],
            'FOM_WIND[USD]':results['FOM_WIND'][0],
            'FOM_EL[USD]':results['FOM_EL'][0],
            'FOM_UG[USD]':results['FOM_UG'][0],
            'H_total[kg]':results['H_total'][0],
            'pv_capacity[kW]': results['pv_max'][0],
            'wind_capacity[kW]': results['wind_max'][0],
            #'pv_capacity_array[kW]': results['pv_max_array'].tolist(),
            #'wind_capacity_array[kW]': results['wind_max_array'].tolist(),
            'el_capacity[kW]': results['el_max'][0],
            'ug_capcaity[kgH2]': results['ug_storage_capa'][0],
            'pipe_storage_capacity[kgH2]': results['pipe_storage_capa'][0],
            'bat_e_capacity[kWh]': results['bat_e_capa'][0],
            'bat_p_capacity[kW]': results['bat_p_max'][0],
            'pv_cost[USD]': results['pv_max'][0] * simparams['C_PV'],
            'wind_cost[USD]': results['wind_max'][0] * simparams['C_WIND'],
            'el_cost[USD]': results['el_max'][0] * simparams['C_EL'],
            'ug_storage_cost[USD]': results['ug_storage_capa'][0] * simparams['C_UG_STORAGE'],
            'pipe_storage_cost[USD]': results['pipe_storage_capa'][0] * simparams['C_PIPE_STORAGE'],
            'bat_cost[USD]': results['bat_p_max'][0] * simparams['C_BAT_ENERGY'],
            'load[kg/s]': results['LOAD'][0],
            'C_trans[USD]': results['C_trans'][0],
            'C_pipe[USD]': results['C_pipe'][0]
        }
        if len(results['pv_max_array'])>1:
            for j in range(len(results['pv_max_array'])):
                row_data['pv_capacity_%s[kW]'%PV_location[j]] = results['pv_max_array'][j]
            for j in range(len(results['pv_max_array'])):
                row_data['wind_capacity_%s[kW]'%PV_location[j]] = results['wind_max_array'][j]
        data_list.append(row_data)
    # Convert list of dictionaries to DataFrame
    RESULTS = pd.DataFrame(data_list)

    #RESULTS
    parent_directory = os.path.dirname(os.getcwd())
    path_to_file = parent_directory + os.sep + 'DATA' + os.sep + 'OPT_OUTPUTS' + os.sep 
    result_file = 'results_2020.csv'

    RESULTS.to_csv(path_to_file+result_file, index=False)
    
def Resource_data(PV_location_g,Coor_PV_x_g,Coor_PV_y_g):
    import glob
    import shutil
    location = 'Tas'
    
    # read the lat and long from existing wind data
    raw_data_folder = datadir + os.sep + 'SAM_INPUTS' + os.sep + 'WEATHER_DATA'+ os.sep + 'raw_data' + os.sep + location
    csv_files = glob.glob(os.path.join(raw_data_folder, 'BARRA*'))
    Latitude = []
    Longitude = []
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        parts = filename.split('-')
        Latitude.append(-float(parts[3])) # - for south hemisphere
        Longitude.append(float(parts[4]))
    Lat = np.linspace(min(Latitude),max(Latitude),int(round((max(Latitude)-min(Latitude))/0.11,0))+1)
    Long = np.linspace(min(Longitude),max(Longitude),int(round((max(Longitude)-min(Longitude))/0.11,0))+1)
   
    for i in range(len(PV_location_g)):
        input_lat = Coor_PV_x_g[i]
        input_lon = Coor_PV_y_g[i]
        # find closest point to the input
        index_lat = np.where(input_lat-Lat<0)[0][0]
        index_lon = np.where(Long-input_lon>0)[0][0]
        Index_lat = np.array([index_lat-1,index_lat-1,index_lat,index_lat])
        Index_lon = np.array([index_lon-1,index_lon,index_lon-1,index_lon])
        List_lat = Lat[Index_lat]
        List_lon = Long[Index_lon]
        distance = np.sqrt((List_lat-input_lat)**2+(List_lon-input_lon)**2)
        
        k = np.where(distance==min(distance))[0][0]
        closest_lat = round(Lat[Index_lat[k]],2)
        closest_long = round(Long[Index_lon[k]],2)
        
        # read raw wind data
        raw_data_file = raw_data_folder + os.sep + 'BARRA-output-%s-%s-2014.csv'%(closest_lat,closest_long)
        try:
            df = pd.read_csv(raw_data_file)
        except:
            print ('Raw data does not match for %s %s'%(input_lat,input_lon))
        
        
        # read raw solar data, now we are using data from ninja
        raw_data_file2 = raw_data_folder + os.sep + 'Process_data_%s_%s_2019.csv'%(closest_lat,closest_long)
        try:
            df_solar = pd.read_csv(raw_data_file2)
        except:
            print ('Raw data does not match for %s %s'%(input_lat,input_lon))
        
        weather_data_folder = datadir + os.sep + 'SAM_INPUTS' + os.sep + 'WEATHER_DATA'
        new_file = weather_data_folder + os.sep + 'weather_data_%s.csv'%PV_location_g[i]
        shutil.copy(weather_data_folder + os.sep + 'weather_data_template.csv', new_file)
        
        df_new = pd.read_csv(weather_data_folder + os.sep + 'weather_data_template.csv')
        
        # change lat, long, wspd, and wdir, DNI and GHI
        df_new.loc[0, 'lat'] = Coor_PV_x_g[i]
        df_new.loc[0, 'lon'] = Coor_PV_y_g[i]
        df_new.loc[2:, 'Snow Depth Units'] = df['wdir'].values
        df_new.loc[2:, 'Pressure Units'] = df['wspd'].values
        df_new.loc[2:, 'Dew Point Units'] = df_solar['DNI'].values
        df_new.loc[2:, 'DNI Units'] = df_solar['GHI'].values
        df_new.to_csv(new_file, index=False)

def load_txt():
    PV_location_g = np.array([])
    Coor_PV_x_g = np.array([])
    Coor_PV_y_g = np.array([])
    El_location_g = np.array([])
    Coor_elx_x_g = np.array([])
    Coor_elx_y_g = np.array([])
    Pipe_buffer = np.array([])
    Area = np.array([])
    inputFileName = os.getcwd()+'/input_solar_test.txt'
    f = open( inputFileName )    
    lines = f.readlines()
    f.close()    
    for line in lines:
        cleanLine = line.strip() 
        if cleanLine[0] == "L" or cleanLine[0] == "#": 
            continue
        elif cleanLine[0] != "#" and cleanLine[0] != "F" :
            splitReturn = splitReturn = cleanLine.split(",")
            PV_location_g = np.append(PV_location_g,[(splitReturn[0])])
            Coor_PV_x_g = np.append(Coor_PV_x_g,[float(splitReturn[1])])
            Coor_PV_y_g = np.append(Coor_PV_y_g,[float(splitReturn[2])])
            Area = np.append(Area,[float(splitReturn[3])])
            Pipe_buffer = np.append(Pipe_buffer,[splitReturn[4]])
            
            El_location_g = np.append(El_location_g,[(splitReturn[0])])
            Coor_elx_x_g = np.append(Coor_elx_x_g,[float(splitReturn[1])])
            Coor_elx_y_g = np.append(Coor_elx_y_g,[float(splitReturn[2])])
        elif cleanLine[0] == "F":
            splitReturn = splitReturn = cleanLine.split(",")
            user_x = float(splitReturn[1])
            user_y = float(splitReturn[2])
            El_location_g = np.append(El_location_g,[(splitReturn[0])])
            Coor_elx_x_g = np.append(Coor_elx_x_g,[float(splitReturn[1])])
            Coor_elx_y_g = np.append(Coor_elx_y_g,[float(splitReturn[2])])
    print (PV_location_g)
    return PV_location_g,Coor_PV_x_g,Coor_PV_y_g,El_location_g,Coor_elx_x_g,Coor_elx_y_g,user_x,user_y,Pipe_buffer,Area
            
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
    return (np.average(capacity_factors_arr))

def solar_output(Location):
    from calendar import monthrange
    
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
    return (capacity_factors_arr)

def solar_assessment():
    # read input file
    data_list = np.array([])
    PV_location_g,Coor_PV_x_g,Coor_PV_y_g,El_location_g,Coor_elx_x_g,Coor_elx_y_g,user_x,user_y,Pipe_buffer,Area = load_txt()
    #data_list = np.append(data_list,PV_location_g[:])
    
    for i in range(1):#len(PV_location_g)):
        Location = PV_location_g[i]
        print (Location)
        SolarResource(Location)
        results = np.average(solar_output(Location))
        data_list = np.append(data_list,results)
    #data_list = data_list.reshape(12,int(len(data_list)/12))
    df = pd.DataFrame(data_list)

    # Save DataFrame to CSV
    df.to_csv('output.csv', index=False)
        
def CF_output():
    PV_location_g,Coor_PV_x_g,Coor_PV_y_g,El_location_g,Coor_elx_x_g,Coor_elx_y_g,user_x,user_y,Pipe_buffer,Area = load_txt()
    
    CF = np.array([])
    for j in range(1):#len(PV_location_g)):
        loc = PV_location_g[j]
        print 
        print (loc)
        update_resource_data(loc)
        solar_output(loc)
        #CF = np.append(CF,[loc,Coor_PV_x_g[j],Coor_PV_y_g[j],round(solar_output(loc),3),round(wind_output(loc),3)])
    
    '''
    CF = CF.reshape(int(len(CF)/5),5)
    df = pd.DataFrame(CF, columns=['Location', 'Lat', 'Long', 'Solar CF', 'Wind CF'])
    df.to_csv(os.getcwd() + os.sep + 'CF_output.txt', sep=',', index=False, header=True)
    '''
if __name__=='__main__':
    optimisation()
    #solar_assessment()
    #plot(location)
    #plot_yearly()
    
    #CF_output()
    #load_txt()