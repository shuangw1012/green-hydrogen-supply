# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:28:44 2022

@author: Ahmad Mojiri
"""
from projdirs import optdir
import numpy as np
from PACKAGE.component_model import pv_gen, wind_gen,SolarResource, WindSource,WindSource_windlab
import os


def make_dzn_file(DT, EL_ETA, BAT_ETA_in, BAT_ETA_out,
                  C_PV, C_WIND, C_EL, C_UG_STORAGE,UG_STORAGE_CAPA_MAX,
                  C_PIPE_STORAGE,PIPE_STORAGE_CAPA_MIN, C_BAT_ENERGY,
                  C_BAT_POWER, OM_PV, OM_WIND, OM_EL, OM_UG,DIS_RATE,
                  CF, PV_REF,WIND_REF,
                  LOAD, C_PV_t, C_wind_t, PV_REF_POUT,WIND_REF_POUT):
    # pdb.set_trace()    
    n_project = 25
    crf = DIS_RATE * (1+DIS_RATE)**n_project/((1+DIS_RATE)**n_project-1)
    # pdb.set_trace()    
    H_total = (CF/100)*sum(LOAD)*DT*3600
    
    string = """
    N = %i;
    n_PV = %i;
    n_wind = %i;
    DT = %.2f;      %% time difference between sample points (hr)
    n_project = %s;
    EL_ETA = %.2f;  %% conversion factor of the electrolyser
    BAT_ETA_in = %.2f;   %%charging efficiency of electrochemical battery
    BAT_ETA_out = %.2f;  %%discharging efficiency of electrochemical battery 
    
    C_PV = %.2f;    %% unit cost of PV ($/kW)
    C_WIND =  %.2f;    %% unit cost of Wind farm ($/kW)
    C_EL =  %.2f;    %% unit cost of electrolyser ($/kW)
    C_UG_STORAGE = %.2f;    %% unit cost of hydrogen storage ($/kgH)
    UG_STORAGE_CAPA_MAX = %.2f; %%maximum size of underground storage $/(kg of H2)
    C_PIPE_STORAGE = %.2f; %% unit cost of storage with line packing $/(kg of H2)
    PIPE_STORAGE_CAPA_MIN = %.2f; %% minimum size of line packing (kg of H2)
    
    C_BAT_ENERGY = %.2f;   %% unit cost of electrochemical battery energy ($/kWh)
    C_BAT_POWER = %.2f;   %% unit cost of electrochemical battery power ($/kWh)
    
    OM_PV = %.2f;    %% Annual O&M cost of PV ($/kW)
    OM_WIND = %.2f;  %% Annual O&M cost of wind ($/kW)
    OM_EL = %.2f;    %% Annual O&M cost of electrolyser ($/kW)
    OM_UG = %.2f;    %% %% Annual O&M cost of underground storage ($/kg)
    
    RES_H_CAPA = %.2f;       %% reserved hydrogen for lowered capcaity factor
    
    PV_REF = %.2f;       %%the capacity of the reference PV plant (kW)
    
    WIND_REF = %.2f;  %% the capacity of the refernce wind plant (kW)
    
    %% load timeseries (kgH/s)                             
    LOAD = %s;
    
    %% Transmission cost (USD/kW)                             
    C_PV_t = %s;
    
    %% Transmission cost (USD/kW)                             
    C_wind_t = %s;
    
    %% discount rate in absolute value not in percentage
    DIS_RATE = %s;
    
    %%capital recovery factor
    crf = %s;
    
    %%Hydrogen production
    H_total = %s;
        
    """ %(len(LOAD), len(PV_REF_POUT), len(WIND_REF_POUT), DT, int(n_project),EL_ETA, BAT_ETA_in, BAT_ETA_out,
      C_PV, C_WIND, C_EL, C_UG_STORAGE, UG_STORAGE_CAPA_MAX, C_PIPE_STORAGE,
      PIPE_STORAGE_CAPA_MIN, C_BAT_ENERGY,
      C_BAT_POWER, OM_PV, OM_WIND, OM_EL, OM_UG, (1-CF/100)*sum(LOAD)*DT*3600, PV_REF, WIND_REF,
      str(LOAD), str(C_PV_t), str(C_wind_t),DIS_RATE,crf,H_total)
    
    with open(optdir + "hydrogen_plant_data_%s.dzn"%(str(CF)), "w") as file:
        file.write(string)
        
        file.write("%% Power output time series from reference PV plant (W)\n")
        file.write("PV_REF_POUT= [")

        # Loop through the rows of the 2D array
        for i,row in enumerate(PV_REF_POUT):
            file.write("|")
            # Loop through the elements of each row
            for j, element in enumerate(row):
                file.write(str(element))
                if j < len(row) - 1 or i!=len(PV_REF_POUT)-1:
                    file.write(", ")
            
            # Close the row
            if i!=len(PV_REF_POUT)-1:
                file.write(" \n")

        file.write(" |];")
        
        file.write(" \n")
        file.write(" \n")
        
        file.write("%% Power output time series from reference Wind plant (W)\n")
        file.write("WIND_REF_POUT= [")

        # Loop through the rows of the 2D array
        for i,row in enumerate(WIND_REF_POUT):
            file.write("|")
            # Loop through the elements of each row
            for j, element in enumerate(row):
                file.write(str(element))
                if j < len(row) - 1 or i!=len(WIND_REF_POUT)-1:
                    file.write(", ")
            
            # Close the row
            if i!=len(WIND_REF_POUT)-1:
                file.write(" \n")

        file.write(" |];")
        
def Minizinc(simparams):
    """
    Parameters
    ----------
    simparams : a dictionary including the following parameters:
        DT, ETA_PV, ETA_EL, C_PV, C_W, C_E, C_HS, CF, pv_ref_capa,
                  W, pv_ref_out, L

    Returns
    -------
    a list of outputs including the optimal values for CAPEX, p-pv, p_w, p_e,
    e_hs

    """
    #make_dzn_file(**simparams)
    
    #mzdir = parent_directory + os.sep + 'MiniZinc'
    # I commented out the mzdir command because it is annoyying to refer to the installation dir
    # in different systems. I think a better way is that we add minizinc to an environment variable 
    #during the installation
    
    minizinc_data_file_name = "hydrogen_plant_data_%s.dzn"%(str(simparams['CF']))
    from subprocess import check_output
    output = str(check_output([#mzdir + 
                               'minizinc', "--soln-sep", '""',
                               "--search-complete-msg", '""', "--solver",
                               "COIN-BC", optdir + "hydrogen_plant.mzn",
                               optdir + minizinc_data_file_name]))
    
    output = output.replace('[','').replace(']','').split('!')
    for string in output:
        if 'CAPEX' in string:
            results = string.split(';')
    
    results = list(filter(None, results))
    
    RESULTS = {}
    for x in results:
        RESULTS[x.split('=')[0]]=np.array((x.split('=')[1]).split(',')).astype(float)        
    
    #remove the minizinc data file after running the minizinc model
    
    #mzfile = optdir + minizinc_data_file_name
    #if os.path.exists(mzfile):
    #    os.remove(mzfile)
    
    
    return(  RESULTS  )

def Optimise(load, cf, storage_type, simparams,PV_location,Wind_location,C_PV_t,C_wind_t):
    simparams.update(CF = cf)

    PV_pv_ref_pout = np.array([])
    Wind_ref_pout = np.array([])
    for loc in PV_location:
        SolarResource(loc)
        
        pv_ref = 1e3 #(kW)
        pv_ref_pout = list(np.trunc(100*np.array(pv_gen(pv_ref)))/100)
        
        PV_pv_ref_pout = np.append(PV_pv_ref_pout,pv_ref_pout)
    PV_pv_ref_pout = PV_pv_ref_pout.reshape(len(PV_location),8760)
    
    
    for loc2 in Wind_location:
        #Update the weather data files
        WindSource_windlab(loc2)
        
        wind_ref = 320e3 #(kW)
        wind_ref_pout = list(np.trunc(100*np.array(wind_gen()))/100)
        Wind_ref_pout = np.append(Wind_ref_pout,wind_ref_pout)
    
    Wind_ref_pout = Wind_ref_pout.reshape(len(Wind_location),8760)
    
    initial_ug_capa = 110
    simparams.update(DT = 1,#[s] time steps
                     PV_REF = pv_ref, #capacity of reference PV plant (kW)
                     WIND_REF = wind_ref, #capacity of reference wind farm (kW)
                     C_UG_STORAGE = Cost_hs(initial_ug_capa, storage_type),
                     LOAD = [load for i in range(len(pv_ref_pout))], #[kgH2/s] load profile timeseries
                     CF = cf,           #capacity factor
                     C_PV_t = C_PV_t,
                     C_wind_t = C_wind_t,
                     PV_REF_POUT = PV_pv_ref_pout,
                     WIND_REF_POUT = Wind_ref_pout
                     )
    
    #print('Calculating for CF=', simparams['CF'])
    #results = Pulp(simparams)
    
    make_dzn_file(**simparams)
    results = Minizinc(simparams)
    
    if simparams['UG_STORAGE_CAPA_MAX']>0:
        new_ug_capa = results['ug_storage_capa'][0]/1e3
        if np.mean([new_ug_capa,initial_ug_capa]) > 0:
            if abs(new_ug_capa - initial_ug_capa)/np.mean([new_ug_capa,initial_ug_capa]) > 0.05:
                initial_ug_capa = new_ug_capa
                print('Refining storage cost; new storage capa=', initial_ug_capa)
                simparams['C_UG_STORAGE'] = Cost_hs(initial_ug_capa, storage_type)
                #results = Pulp(simparams)
                make_dzn_file(**simparams)
                results = Minizinc(simparams)
    
    results.update(CF=simparams['CF'],
                   C_UG_STORAGE=simparams['C_UG_STORAGE'])
    
    return(results,simparams)
    

def Pulp(simparams):   
    from pulp import LpVariable,LpProblem,LpMinimize,LpStatus
    import pulp
    # pass on the parameters
    N = len(simparams['PV_REF_POUT']) # number of hours
    time_steps = range(0, N)
    EL_ETA = simparams['EL_ETA']
    BAT_ETA_in = simparams['BAT_ETA_in']
    BAT_ETA_out = simparams['BAT_ETA_out']
    C_PV = simparams['C_PV']
    C_WIND = simparams['C_WIND']
    C_EL = simparams['C_EL']
    UG_STORAGE_CAPA_MAX = simparams['UG_STORAGE_CAPA_MAX']
    C_PIPE_STORAGE = simparams['C_PIPE_STORAGE']
    PIPE_STORAGE_CAPA_MIN = simparams['PIPE_STORAGE_CAPA_MIN']
    C_BAT_ENERGY = simparams['C_BAT_ENERGY']
    C_BAT_POWER = simparams['C_BAT_POWER']
    DT = simparams['DT']
    PV_REF = simparams['PV_REF']
    PV_REF_POUT = simparams['PV_REF_POUT']
    WIND_REF = simparams['WIND_REF']
    WIND_REF_POUT = simparams['WIND_REF_POUT']
    C_UG_STORAGE = simparams['C_UG_STORAGE']
    LOAD = simparams['LOAD']
    CF = simparams['CF']
    RES_H_CAPA = C_BAT_POWER,(1-CF/100)*sum(LOAD)*DT*3600
    
    # Initialize Class
    prob = LpProblem('Green_Hydrogen_Supply_Optimization', LpMinimize)
    
    # Create Decision Variables
    pv_max = LpVariable('pv_max', 0, None) # PV plant rated power (kW)
    wind_max = LpVariable('wind_max', 0, None) # wind plant rated power (kW)
    el_max = LpVariable('el_max', 0, None) # electrolyser rated power (kW)
    ug_storage_capa = LpVariable('ug_storage_capa', 0, None) # capacity of hydrogen storage cavern (kg of H2)
    pipe_storage_capa = LpVariable('pipe_storage_capa', 0, None) # capacity of hydrogen storage in the pipeline (kg of H2)
    bat_e_capa = LpVariable('bat_e_capa', 0, None) # energy capacity of the electrochemical battery (kWh)
    bat_p_max = LpVariable('bat_p_max', 0, None) # power capacity of the electrochemical battery (kW)
    
    pv_pout = LpVariable.dicts("pv_pout", (time_steps), lowBound=0) # power out of PV plant (kW)
    wind_pout = LpVariable.dicts("wind_pout", (time_steps), lowBound=0) # power out of wind farm (kW)
    curtail_p = LpVariable.dicts("curtail_p", (time_steps), lowBound=0) # curtailed power (kW)
    el_pin_pvwind = LpVariable.dicts("el_pin_pvwind", (time_steps), lowBound=0) # power from wind and pv into the electrolyser (kW)
    res_hout = LpVariable.dicts("res_hout", (time_steps), lowBound=0) # hydrogen extracted from virtual reserve (kgH/s)
    comp1_hflow = LpVariable.dicts("comp1_hflow", (time_steps), lowBound=0) # hydrogen flowing into compressor 1 (kg of H2/s)
    comp1_pin = LpVariable.dicts("comp1_pin", (time_steps), lowBound=0) # power into compressor 1 (kW)
    comp2_pin = LpVariable.dicts("comp2_pin", (time_steps), lowBound=0) # power into compressor 2 (kW)
    el_pin = LpVariable.dicts("el_pin", (time_steps), lowBound=0) # power flow into the electrolyser (kW)
    bat_pin = LpVariable.dicts("bat_pin", (time_steps), lowBound=0) # power flow into the battery (kW)
    bat_pout = LpVariable.dicts("bat_pout", (time_steps), lowBound=0) # power flow out of the battery (kW)
    comp2_hflow = LpVariable.dicts("comp2_hflow", (time_steps), lowBound=0) # hydrogen transfer from pipeline to underground storage (kg/s)
    pipe_storage_hout = LpVariable.dicts("pipe_storage_hout", (time_steps), lowBound=0) # hydrogen flow from the pipe storage to the load (kg of H2/s)
    ug_storage_hout = LpVariable.dicts("ug_storage_hout", (time_steps), lowBound=0) # discharge from underground storage (kg of H2/s)
    ug_storage_level = LpVariable.dicts("ug_storage_level", (time_steps), lowBound=0) # stored hydrogen level in underground storage (kg)
    pipe_storage_level = LpVariable.dicts("pipe_storage_level", (time_steps), lowBound=0) # stored hydrogen level in pieplie (kg)
    res_h_level = LpVariable.dicts("res_h_level", (time_steps), lowBound=0) # reserved hydrogen for load shut down (kg)
    bat_e = LpVariable.dicts("bat_e", (time_steps), lowBound=0) # electrical energy stored in the battery (kWh)
    
    # The objective function is added to 'prob'
    capex = C_PV * pv_max + C_WIND * wind_max + C_EL * el_max + C_UG_STORAGE * ug_storage_capa + C_PIPE_STORAGE * pipe_storage_capa + C_BAT_ENERGY * bat_e_capa + C_BAT_POWER * bat_p_max
    prob += capex
    
    # Constraints
    prob += res_h_level[0] == RES_H_CAPA
    prob += ug_storage_level[0] == ug_storage_level[N-1]
    prob += pipe_storage_level[0] == pipe_storage_level[N-1]
    prob += bat_e[0] == bat_e[N-1]
    prob += ug_storage_capa <= UG_STORAGE_CAPA_MAX
    prob += pipe_storage_capa >= PIPE_STORAGE_CAPA_MIN
    i=N-1
    
    for i in time_steps:
        prob += PV_REF * pv_pout[i] == pv_max * PV_REF_POUT[i]
        prob += WIND_REF * wind_pout[i] == wind_max * WIND_REF_POUT[i]
        prob += pv_pout[i] + wind_pout[i] - curtail_p[i] - el_pin_pvwind[i] - bat_pin[i] == 0
        prob += curtail_p[i] >= 0
        prob += el_pin[i] == el_pin_pvwind[i] + bat_pout[i] - comp1_pin[i] - comp2_pin[i]
        prob += bat_pin[i] >= 0
        prob += bat_pin[i] - bat_p_max <= 0
        prob += bat_pout[i] >= 0
        prob += bat_pout[i] - bat_p_max <= 0
        prob += el_pin_pvwind[i] >= 0
        prob += el_pin_pvwind[i] + bat_pout[i] - comp1_pin[i] - comp2_pin[i] - el_max <= 0
        if i != N-1:
            prob += bat_e[i+1] == bat_e[i] + ( bat_pin[i] * BAT_ETA_in - bat_pout[i]*1/BAT_ETA_out ) * DT
        prob += bat_e[i] >= 0
        prob += bat_e[i] - bat_e_capa <= 0
        prob += ug_storage_level[i] >= 0
        prob += ug_storage_level[i] - ug_storage_capa <= 0
        prob += pipe_storage_level[i] >= 0
        prob += pipe_storage_level[i] - pipe_storage_capa <= 0
        prob += comp1_pin[i] == comp1_hflow[i] * 0.83 * 3600
        prob += comp2_pin[i] == comp2_hflow[i] * 0.41 * 3600
        prob += comp1_hflow[i] == el_pin[i]*1/3600* EL_ETA*1/ 39.4 # high calorific value of H2 = 39.4 kWh/kg
        prob += comp2_hflow[i] >= 0
        prob += comp2_hflow[i] <= el_max*1/3600* EL_ETA*1/ 39.4
        if i != N-1:
            prob += ug_storage_level[i+1] == ug_storage_level[i] + (comp2_hflow[i] - ug_storage_hout[i] )*DT*3600
            prob += pipe_storage_level[i+1] == pipe_storage_level[i] + (comp1_hflow[i] - pipe_storage_hout[i] - comp2_hflow[i]) *DT*3600
        prob += pipe_storage_hout[i] + ug_storage_hout[i] + res_hout[i] == LOAD[i]
        prob += pipe_storage_hout[i] >= 0
        prob += ug_storage_hout[i] >= 0
        prob += res_h_level[i] >= 0
        prob += res_h_level[i] <= RES_H_CAPA 
        if i != N-1:
            prob += res_h_level[i+1] == res_h_level[i] - res_hout[i]*DT*3600
        prob += res_hout[i] >= 0
        prob += res_hout[i] <= LOAD[i]
        
    # The problem data is written to an .lp file
    prob.writeLP("optimisation.lp")

    # The problem is solved using PuLP's choice of Solver
    import time
    start = time.time()
    solver = pulp.PULP_CBC_CMD(msg=1,gapRel=10,timeLimit=100)
    prob.solve()
    print ('simulated time', time.time()-start)
    # The status of the solution is printed to the screen
    #print("Status:", LpStatus[prob.status])
        
    solution_dict = {}
    for var in prob.variables():
        solution_dict[var.name] = var.varValue
        
    # Specify the CSV file path
    csv_file_path = "solution_%s.csv"%CF
    import csv
    # Write solution data to the CSV file
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Variable', 'Value'])
        for var_name, var_value in solution_dict.items():
            csv_writer.writerow([var_name, var_value])
    
    capex = C_PV * solution_dict['pv_max'] + C_WIND * solution_dict['wind_max'] + C_EL * solution_dict['el_max'] + C_UG_STORAGE * solution_dict['ug_storage_capa'] +  C_PIPE_STORAGE * solution_dict['pipe_storage_capa'] + C_BAT_ENERGY * solution_dict['bat_e_capa'] + C_BAT_POWER * solution_dict['bat_p_max']
    
    results_dict = {}
    results_dict['N'] = N
    results_dict['CAPEX'] = [capex]
    results_dict['pv_max'] = [solution_dict['pv_max']]
    results_dict['wind_max'] = [solution_dict['wind_max']]
    results_dict['el_max'] = [solution_dict['el_max']]
    results_dict['ug_storage_capa'] = [solution_dict['ug_storage_capa']]
    results_dict['pipe_storage_capa'] = [solution_dict['pipe_storage_capa']]
    results_dict['bat_e_capa'] = [solution_dict['bat_e_capa']]
    results_dict['bat_p_max'] = [solution_dict['bat_p_max']]
    #results_dict['pv_pout'] = solution_dict['pv_pout']
    #results_dict['wind_pout'] = solution_dict['wind_pout']
    #results_dict['curtail_p'] = solution_dict['curtail_p']
    #results_dict['bat_pin'] = solution_dict['bat_pin']
    #results_dict['bat_pout'] = solution_dict['bat_pout']
    #results_dict['el_pin'] = solution_dict['el_pin']
    #results_dict['comp1_hflow'] = solution_dict['comp1_hflow']
    #results_dict['comp1_pin'] = solution_dict['comp1_pin']
    #results_dict['comp2_hflow'] = solution_dict['comp2_hflow']
    #results_dict['comp2_pin'] = solution_dict['comp2_pin']
    #results_dict['res_hout'] = solution_dict['res_hout']
    #results_dict['pipe_storage_hout'] = solution_dict['pipe_storage_hout']
    #results_dict['ug_storage_hout'] = solution_dict['ug_storage_hout']
    #results_dict['pipe_storage_level'] = solution_dict['pipe_storage_level']
    #results_dict['ug_storage_level'] = solution_dict['ug_storage_level']
    #results_dict['reserve_h_level'] = solution_dict['reserve_h_level']
    #results_dict['bat_e'] = solution_dict['bat_e']
    #results_dict[''] = solution_dict['']
    results_dict['LOAD'] = LOAD
    #print (results_dict)
    return results_dict
    
    
def Cost_hs(size,storage_type):
    """
    This function calculates the unit cost of storage as a function of size
    
    Parameters
    ----------
    size: storage capacity in kg of H2
    storage_type: underground storage type; 
                one of ['Lined Rock', 'Salt Cavern']

    Returns unit cost of storage in USD/kg of H2
        
    """
    if size > 0:
        x = np.log10(size)
        if size > 100:
            if storage_type == 'Salt Cavern':
                cost=10 ** (0.212669*x**2 - 1.638654*x + 4.403100)
                if size > 8000:
                    cost = 17.66
            elif storage_type == 'Lined Rock':
                cost =10 ** (   0.217956*x**2 - 1.575209*x + 4.463930  )
                if size > 4000:
                    cost = 41.48
        else:
            cost = 10 ** (-0.0285*x + 2.7853)
    else:
        cost = 516
    return(cost)