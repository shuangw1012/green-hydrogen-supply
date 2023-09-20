# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 18:54:47 2022

@author: Ahmad Mojiri
"""

from projdirs import optdir
import numpy as np
import pandas as pd
import platform



def read_data_for_plotting(results):
        data_to_plot = {}
        (KEYS,VALUES) = (list(results.keys()),list(results.values()))
        for (k,v) in zip(KEYS,VALUES):
            if len(v)>1:
                data_to_plot[k]=v[0:int(results['N'][0])]
        return(pd.DataFrame(data_to_plot))



def prep_results_to_print(results,simparams):
    
    RESULTS =  {'Capacity Factor': results['CF'],
                'CAPEX [USD]':results['CAPEX'][0],
                'PV Rated Power[kW]': results['pv_max'][0],
                'Wind Rated Power [kW]': results['wind_max'][0],
                'Electrolyser rated Power [kW]': results['el_max'][0],
                'UG H2 Capacity [kgH2]': results['ug_storage_capa'][0],
                'Pipe Storage Capacity [kgH2]': results['pipe_storage_capa'][0],
                'Battery Energy Capacity [kWh]': results['bat_e_capa'][0],
                'Battery Power Capacity [kW]': results['bat_p_max'][0],
                'PV Cost [USD]': results['pv_max'][0]*simparams['C_PV'],
                'Wind Cost [USD]': results['wind_max'][0]*simparams['C_WIND'],
                'Electrolyser Cost [USD]': results['el_max'][0]*simparams['C_EL'],
                'UG Storage Cost [USD]': results['ug_storage_capa'][0]*results['C_UG_STORAGE'],
                'Pipe Storage Cost [USD]':results['pipe_storage_capa'][0]*simparams['C_PIPE_STORAGE'],
                'Battery Cost [USD]': results['bat_p_max'][0]*simparams['C_BAT_ENERGY'],
                'Load [kg/s]':results['LOAD'][0],
                'UG Storage Unit Price [USD/kgH2]':results['C_UG_STORAGE']
                 }
    # output = str(RESULTS).replace(', ',',\n ').replace('{', '').replace('}', '').replace("'",'').replace(',','')   
    return(RESULTS)





def prep_results_to_plot(results,simparams,Location):
    del results['CF']
    del results['C_UG_STORAGE']
    
    if platform.system()=='Linux':
        path = r'/home/ahmadmojiri/GreenH2/DATA/SAM_INPUTS/WEATHER_DATA/'
    else:
        path = r'C:\Nextcloud\HILT-CRC---Green-Hydrogen\DATA\SAM_INPUTS\WEATHER_DATA\\'
        
    weather_data = pd.read_csv(path + 'weather_data_%s.csv'%(Location),skiprows=2)
    
    
    
    
    data_plot = read_data_for_plotting(results)
    
    beam = weather_data['DNI'].tolist()
    G_H = weather_data['GHI']
    wind_speed = weather_data['Wind Speed']
    
    water_rate = 15 #L/kgH2  range: 15-20 L/kgH2
    c_water = 5 # $/m3
    
    data_plot['beam'] = beam
    data_plot['t'] = np.arange(0,results['N'])
    data_plot['time'] =pd.to_datetime('2021-01-01') + pd.to_timedelta(data_plot.t, 'h')
    data_plot.drop('t',axis=1,inplace=True)
    data_plot['G_H'] = G_H
    data_plot['wind_speed'] = wind_speed   
    return(data_plot)
    
    




def crf(rate,years):
    return(  (rate*(1+rate)**years) / ((1+rate)**years-1)  )




# def LCOH2(rate, life, cf, load, pv_fom, wind_fom, elec_fom,
#           pv_vom, wind_vom, elec_vom):
    




    