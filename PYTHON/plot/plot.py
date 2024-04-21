import matplotlib.pyplot as plt
import numpy as np

import os

plt.rcParams["font.family"] = "Times New Roman"
#from projdirs import datadir
import pandas as pd

def plot_bar():
    n_project = 25
    DIS_RATE = 0.06
    crf = DIS_RATE * (1+DIS_RATE)**n_project/((1+DIS_RATE)**n_project-1)
    LOAD = 2.115
    CF = 100
    H_total = (CF/100)*LOAD*8760*3600
    
    Filter_location = ['Burnie_new1','Burnie_new2','Burnie_new3','Burnie_new4','Burnie_new5','Burnie_new6','User']
    #Filter_location = ['Optimal']
    
    for filter_location in Filter_location:
        dir = datadir + os.sep + 'OPT_OUTPUTS' + os.sep
        file_name = 'results_2020_10-10.csv'
        df = pd.read_csv(dir+file_name)
        data = df[df['El'] == filter_location]
        #data = df[df['Candidate'] == filter_location]
        data = data.reset_index(drop=True)
    
        
        cases = ['Burnie 1', 'Burnie 2', 'Burnie 3', 'Burnie 4', 'Burnie 5', 'Burnie 6', 'User']
        #cases = ['Burnie 1\n LCOH: 3.32', 'Burnie 2\n 3.18', 'Burnie 3\n 2.77', 'Burnie 4\n 2.92', 
        #         'Burnie 5 \n 2.93', 'Burnie 6\n 2.77', 'End user \n 2.96']
        cost_categories = ['Capex-PV', 'Capex-Wind', 'Capex-El', 'Capex-storage', 'Capex-Trans','Capex-Pipe',
                           'FOM-PV','FOM-Wind','FOM-El','FOM-storage']
        lcoh_values = {}
        cost_values = {}
        
        for i in range(len(data['cf'])):
            lcoh_values[cases[i]] = [round(crf*data['pv_cost[USD]'][i]/H_total,2),round(crf*data['wind_cost[USD]'][i]/H_total,2),
                                     round(crf*data['el_cost[USD]'][i]/H_total,2),round(crf*data['ug_storage_cost[USD]'][i]/H_total,2),
                                     round(crf*data['C_trans[USD]'][i]/H_total,2),round(crf*data['C_pipe[USD]'][i]/H_total,2),
                                     round(data['FOM_PV[USD]'][i]/H_total,2),round(data['FOM_WIND[USD]'][i]/H_total,2),
                                     round(data['FOM_EL[USD]'][i]/H_total,2),round(data['FOM_UG[USD]'][i]/H_total,2)]
            
            cost_values[cases[i]] = [int(data['pv_cost[USD]'][i]/1e6),int(data['wind_cost[USD]'][i]/1e6),
                                     int(data['el_cost[USD]'][i]/1e6),int(data['ug_storage_cost[USD]'][i]/1e6),
                                     int(data['C_trans[USD]'][i]/1e6),int(data['C_pipe[USD]'][i]/1e6)]
        
        colors = ['lightblue', 'lightpink', 'lightgray', 'lightgreen', 'lightyellow', 'orange', 'blue', 'magenta', 'brown', 'pink']
        Values = lcoh_values
        colors = colors[:len(Values[cases[0]])]
        cost_categories = cost_categories[:len(Values[cases[0]])]
        
        
        # Create a stacked bar plot
        fig = plt.figure(figsize=(8, 5))
        bottom = np.zeros(len(cases))  # Initialize the bottom of each section
        Bottom = np.array([])
        
        for i, category in enumerate(cost_categories):
            values = [Values[case][i] for case in cases]
            plt.bar(cases, values, label=category, bottom=bottom, color=colors[i],width = 0.5)
            bottom += values  # Update the bottom for the next category
            Bottom = np.append(Bottom,bottom)
        Bottom = Bottom.reshape(len(cost_categories),int(len(Bottom)/len(cost_categories)))
        Bottom = [[row[i] for row in Bottom] for i in range(len(Bottom[0]))]
        
        #plt.ylabel('Cost (MUSD)',fontsize = 14)
        
        plt.ylabel('LCOH (USD/kg)',fontsize = 14)
        
        plt.legend(loc='upper right',ncols=3)
        plt.ylim(0,4.2)
        # Display the numbers on top of the bars with adjusted positions
        
        if Values == cost_values:
            for case in cases:
                bottom = Bottom[cases.index(case)]
                for i, category in enumerate(cost_categories):
                    value = Values[case][i]
                    if value > 0 and i!=5:
                        plt.text(case, bottom[i]-value, str(value), ha='center', va='bottom', fontsize=10,weight='bold')
                        #print (case,i,value,bottom[i])
                    elif i == 5:
                        plt.text(case, bottom[i]+50, str(value), ha='center', va='bottom', fontsize=10,weight='bold')
                        
        plt.tight_layout()
        plt.tick_params(axis='both', labelsize=12)
        #plt.savefig(os.getcwd()+'/Cost_breakdown_%s.png'%filter_location,dpi=100)
        plt.savefig(os.getcwd()+'/lcoh-%s.png'%filter_location,dpi=100)
        plt.close(fig)

def plot_bar_capcity():
    
    file_name = os.getcwd()+'/results_2020_salt1-sorted.csv'
    df = pd.read_csv(file_name)
    
    pv_capacity = df['pv_capacity[kW]'].values/1000
    wind_capacity = df['wind_capacity[kW]'].values/1000
    X = np.linspace(0,len(pv_capacity)-1,len(pv_capacity))
    
    # Create a stacked bar plot
    fig = plt.figure(figsize=(8, 5))
    bottom = np.zeros(len(X))  # Initialize the bottom of each section
    Bottom = np.array([])
    for i in range(len(X)):
        if i == 0:
            plt.bar(X[i], pv_capacity[i], label='pv', bottom=bottom, color='red',width = 0.5)
            plt.bar(X[i], wind_capacity[i], label='wind', bottom=pv_capacity[i], color='blue',width = 0.5)
        else:
            plt.bar(X[i], pv_capacity[i], bottom=bottom, color='red',width = 0.5)
            plt.bar(X[i], wind_capacity[i], bottom=pv_capacity[i], color='blue',width = 0.5)
            
    plt.ylabel('Capacity (MW)',fontsize = 14)
    plt.legend(fontsize =14)
    plt.xlabel('Cases, LCOH increases from left to right',fontsize = 14)
    plt.tick_params(axis='both', labelsize=12)
    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.getcwd()+'/lcoh.png',dpi=300)
    plt.close(fig)

#plot_bar_capcity()

def plot_bar2():
    n_project = 25
    DIS_RATE = 0.06
    crf = DIS_RATE * (1+DIS_RATE)**n_project/((1+DIS_RATE)**n_project-1)
    LOAD = 2.115
    CF = 100
    H_total = (CF/100)*LOAD*8760*3600
    
    #Filter_location = ['Low cost','High cost']
    #Filter_location = ['KF249-Low cost','KF249-High cost','KF249-High cost-2','KJ256-Low cost',
    #                   'KJ256-High cost','KJ256-High cost-2','User-Low cost','User-High cost','User-High cost-2']
    #Filter_location = ['Low-cost, CF100','Low-cost, CF90','High-cost, CF100','High-cost, CF90']
    #Filter_location = ['Salt-100','Salt-90','Rock-100','Rock-90', 'Pipe-100', 'Pipe-90', 'Depleted-100', 'Depleted-90']
    Filter_location = ['Salt Cavern','Lined Rock','Pipeline','Depleted gas field']
    cost_categories = ['PV','Wind', 'Electrolyser', 'Storage', 'Transmission', 'Transportation']
  
    data_multi = np.array([[0,1.98,1.41,0.11,0.02,0.24],
                           [0,1.95,1.39,0.17,0.02,0.05],
                           [0,2.45,1.67,0.46,0.07,0.01],
                           [0,1.98,1.41,0.05,0.02,0.09]])
    
    data = data_multi
    
    colors = ['orange','lightblue', 'lightpink', 'gray', 'lightgreen', 'magenta']#, 'orange', 'blue', 'magenta', 'brown', 'pink']
    
    # Create a stacked bar plot
    fig = plt.figure(figsize=(8, 5))
    bottom = np.zeros(len(Filter_location))  # Initialize the bottom of each section
    Bottom = np.array([])
    
    for i, category in enumerate(cost_categories):
        values = data[:,i]
        print (values)
        plt.bar(Filter_location, values, label=category, bottom=bottom, color=colors[i],width = 0.5)
        bottom += values  # Update the bottom for the next category
        Bottom = np.append(Bottom,bottom)
    Bottom = Bottom.reshape(len(cost_categories),int(len(Bottom)/len(cost_categories)))
    Bottom = [[row[i] for row in Bottom] for i in range(len(Bottom[0]))]
    #plt.ylabel('Cost (MUSD)',fontsize = 14)
    plt.ylabel('LCOH (USD/kg)',fontsize = 14)
    
    plt.legend(loc='upper left',ncols=2)
    plt.ylim(0,5.5)
    
    for j in range(len(Filter_location)):
        bottom = Bottom[j]
        for i in range(len(cost_categories)):
            if data[j,i]==0:
                continue
            
            if i==3:
                plt.text(Filter_location[j], Bottom[j][i]-1.*data[j,i], str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
            elif i==4:
                plt.text(Filter_location[j], Bottom[j][i]+0.12, str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
            elif i==5:
                plt.text(Filter_location[j], Bottom[j][i]+0.32, str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
            else:
                plt.text(Filter_location[j], Bottom[j][i]-0.7*data[j,i], str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
        #plt.text(Filter_location[j], bottom[i], str(round(sum(data[j,:]),2)), ha='center', va='bottom', fontsize=10,weight='bold')
    
    plt.tick_params(axis='both', labelsize=12)
    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.getcwd()+'/lcoh.png',dpi=100)
    plt.close(fig)

#plot_bar2()

def plot_bar22():
    n_project = 25
    DIS_RATE = 0.06
    crf = DIS_RATE * (1+DIS_RATE)**n_project/((1+DIS_RATE)**n_project-1)
    LOAD = 2.115
    CF = 100
    H_total = (CF/100)*LOAD*8760*3600
    
    #Filter_location = ['Low cost','High cost']
    #Filter_location = ['KF249-Low cost','KF249-High cost','KF249-High cost-2','KJ256-Low cost',
    #                   'KJ256-High cost','KJ256-High cost-2','User-Low cost','User-High cost','User-High cost-2']
    #Filter_location = ['Low-cost, CF100','Low-cost, CF90','High-cost, CF100','High-cost, CF90']
    #Filter_location = ['Salt-100','Salt-90','Rock-100','Rock-90', 'Pipe-100', 'Pipe-90', 'Depleted-100', 'Depleted-90']
    Filter_location = ['Salt Cavern\nmine','Salt Cavern\nport',
                       'Lined Rock\nmine','Lined Rock\nport',
                       'Pipe Storage\nmine','Pipe Storage\nport']
    cost_categories = ['PV','Wind', 'Electrolyser', 'Storage', 'Transmission', 'Transportation','Battery']
  
    data_multi = np.array([[0.58,1.62,1.28,0.08,0.01,0.27,0],	
        [2.54,0.00,2.31,0.19,0.00,0.25,0],
        [0.54,1.64,1.30,0.14,0.01,0.02,0],
        [2.56,0.00,2.30,0.34,0.00,0.01,0],
        [0.93,1.56,1.24,0.52,0.05,0.04,0],
        [3.23,0.00,2.44,1.63,0.00,0.02,0.02]])

    data = data_multi
    
    colors = ['orange','lightblue', 'lightpink', 'gray', 'lightgreen', 'magenta', 'brown']#, 'blue', 'magenta', 'brown', 'pink']
    
    # Create a stacked bar plot
    fig = plt.figure(figsize=(8, 5))
    bottom = np.zeros(len(Filter_location))  # Initialize the bottom of each section
    Bottom = np.array([])
    
    for i, category in enumerate(cost_categories):
        values = data[:,i]
        print (values)
        plt.bar(Filter_location, values, label=category, bottom=bottom, color=colors[i],width = 0.5)
        bottom += values  # Update the bottom for the next category
        Bottom = np.append(Bottom,bottom)
    Bottom = Bottom.reshape(len(cost_categories),int(len(Bottom)/len(cost_categories)))
    Bottom = [[row[i] for row in Bottom] for i in range(len(Bottom[0]))]
    #plt.ylabel('Cost (MUSD)',fontsize = 14)
    plt.ylabel('LCOH (USD/kg)',fontsize = 14)
    
    plt.legend(loc='upper left',ncols=3)
    plt.ylim(0,8)
    
    for j in range(len(Filter_location)):
        bottom = Bottom[j]
        for i in range(len(cost_categories)):
            if data[j,i]==0:
                continue
            
            if i==3:
                plt.text(Filter_location[j], Bottom[j][i]-1.*data[j,i], str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
            elif i==4:
                plt.text(Filter_location[j], Bottom[j][i]+0.12, str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
            elif i==5:
                plt.text(Filter_location[j], Bottom[j][i]+0.32, str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
            else:
                plt.text(Filter_location[j], Bottom[j][i]-0.7*data[j,i], str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
        #plt.text(Filter_location[j], bottom[i], str(round(sum(data[j,:]),2)), ha='center', va='bottom', fontsize=10,weight='bold')
    
    plt.tick_params(axis='both', labelsize=12)
    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.getcwd()+'/lcoh.png',dpi=300)
    plt.close(fig)

#plot_bar22()

def plot_GIS_stg():
    import geopandas as gpd
    from shapely.geometry import Point,Polygon,LineString
    import matplotlib.pyplot as plt
    import folium
    from folium.features import DivIcon
    # Load your data from the CSV file
    Results = pd.read_csv(os.path.join(os.getcwd(), 'results_2020_depleted_opt.csv'))
    el_capacity = int(Results['el_capacity[kW]']/1000)
    stg_capacity = max(int(Results['ug_capcaity[kgH2]']/1000),int(Results['pipe_storage_capacity[kgH2]']/1000))
    data = pd.read_csv(os.path.join(os.getcwd(), 'input_Kwinana.txt'))#'input_usg_plot_salt1.txt'))
    notation = False
    for k in range(len(Results['El'].values)):
        #if data['#Name'][k]!= 'KF249':
        #    continue
        Plot_results = np.array([])
        for j in range(len(data['#Name'])-2):
            try:
                if Results['wind_capacity_%s[kW]'%data['#Name'][j]][k]>5000 or Results['pv_capacity_%s[kW]'%data['#Name'][j]][k]>5000:
                    Plot_results=np.append(Plot_results,[data['#Name'][j],data['Lat'][j],data['Long'][j],
                                                         int(Results['wind_capacity_%s[kW]'%data['#Name'][j]][k]/1000),
                                                         int(Results['pv_capacity_%s[kW]'%data['#Name'][j]][k]/1000)
                                                         ])
            except:
                #if data['#Name'][j]!= 'KF249':
                #    continue
                Plot_results=np.append(Plot_results,[data['#Name'][j],data['Lat'][j],data['Long'][j],
                                                     int(Results['wind_capacity[kW]'][k]/1000),
                                                     int(Results['pv_capacity[kW]'][k]/1000)
                                                     ])
        Plot_results = Plot_results.reshape(int(len(Plot_results)/5),5)
        
        el_lat = data[data['#Name']==Results['El'][k]]['Lat'].iloc[0]
        el_long = data[data['#Name']==Results['El'][k]]['Long'].iloc[0]
        
        df = pd.DataFrame(Plot_results, columns=['#Name', 'Lat', 'Long','K_wind','K_PV'])
        
        #df = df.append({'#Name': 'Second user', 'Lat': -20.32, 'Long': 118.60}, ignore_index=True)
        df = df.append({'#Name': 'Electrolyser', 'Lat': el_lat, 'Long': el_long}, ignore_index=True)
        df = df.append({'#Name': 'End user', 'Lat': data['Lat'].values[-1], 'Long': data['Long'].values[-1]}, ignore_index=True)
        df = df.append({'#Name': 'Storage', 'Lat': data['Lat'].values[-2], 'Long': data['Long'].values[-2]}, ignore_index=True)
        
        
        crs = {'init': 'epsg:4326'}
        from shapely.geometry import Point
        
        geometry = [Point(xy) for xy in zip(df["Long"], df["Lat"])]
        geodata = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
        if geodata.iloc[-1]['geometry'] == Point(0,0):
            point_x = (geodata.iloc[-3]['geometry'].x + geodata.iloc[-2]['geometry'].x)/2
            point_y = (geodata.iloc[-3]['geometry'].y + geodata.iloc[-2]['geometry'].y)/2
            geodata.at[len(geodata) - 1, 'geometry'] = Point(point_x, point_y)
        
        # (geodata.iloc[-3]['geometry']) is el
        # (geodata.iloc[-2]['geometry']) is end user
        # (geodata.iloc[-1]['geometry']) is storage
        
        # Create a Folium map
        centre_lat = -32.5#,-22.3#-20.5#-28.11#,
        centre_long = 116.13#119.5#119.1# 119.3#140.23#
        zoom = 8#10#9
        #m = folium.Map(location=[-30, 139], zoom_start=7)
        m = folium.Map(location=[centre_lat, centre_long], zoom_start=zoom, control_scale=True)
        '''
        # Add the connection line (if needed)
        for i in range(len(df)-2):
            line = LineString([geodata.iloc[i]['geometry'], geodata.iloc[-3]['geometry']])
            line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
            #folium.GeoJson(line_gdf).add_to(m)
            line_style = {
            'color': 'red',  # Change the line color (e.g., to red)
            'weight': 3,  # Change the line width
            }
            folium.GeoJson(line_gdf, style_function=lambda x: line_style).add_to(m)
        '''
        # for pipe line
        line = LineString([geodata.iloc[-3]['geometry'], geodata.iloc[-1]['geometry']])
        line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
        line_style2 = {
        'color': 'black',  # Line color
        'weight': 3,  # Line width
        'dashArray': '5, 10',  # Length of dashes and gaps (5 pixels, 10 pixels)
        'dashOffset': '0',}
        folium.GeoJson(line_gdf, style_function=lambda x: line_style2).add_to(m)
        
        line = LineString([geodata.iloc[-3]['geometry'], geodata.iloc[-2]['geometry']])
        line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
        line_style2 = {
        'color': 'black',  # Line color
        'weight': 3,  # Line width
        'dashArray': '5, 10',  # Length of dashes and gaps (5 pixels, 10 pixels)
        'dashOffset': '0',}
        folium.GeoJson(line_gdf, style_function=lambda x: line_style2).add_to(m)
        icon_size = (40,40)
        fontsize = 12
        # Add your original GeoDataFrame (points) to the map with labels
        for i, row in geodata.iterrows():
            if row['#Name'] != 'Electrolyser' and row['#Name'] != 'End user' and row['#Name'] != 'Storage':
                continue
                if row['geometry'].y == el_lat and row['geometry'].x == el_long:
                    delta_x = 0.0
                    delta_y = -0.05
                else:
                    delta_x = 0.05
                    delta_y = 0

                if float(geodata['K_PV'][i]) > 0.1 and float(geodata['K_wind'][i]) > 0.1:
                    pv_icon = folium.features.CustomIcon('%s/Icon/solar-power.png'%os.getcwd(), icon_size=icon_size)
                    folium.Marker(
                        location=[row['geometry'].y+0.015+delta_y, row['geometry'].x-0.015+delta_x],
                        icon=pv_icon,
                        popup=row['#Name'],
                    ).add_to(m)
                    turbine_icon = folium.features.CustomIcon('%s/Icon/wind-power.png'%os.getcwd(), icon_size=icon_size)
                    folium.Marker(
                        location=[row['geometry'].y-0.015+delta_y, row['geometry'].x+0.015+delta_x],
                        icon=turbine_icon,
                        popup=row['#Name'],
                    ).add_to(m)
    
                    if notation == True:
                        character = row['#Name'] + ' ' + '%s/%sMW'%(df['K_PV'][i],df['K_wind'][i])
                        folium.Marker(
                            location=[row['geometry'].y-0.03+delta_y, row['geometry'].x-0.03+delta_x],
                            icon=DivIcon(
                            html='<<div style="font-size: %spt"><b>%s</b></div>' % (fontsize,character),
                                ),
                            popup=row['#Name'],  # Use the "Name" column as the label
                        ).add_to(m)
                
                elif float(geodata['K_PV'][i]) < 0.1 and float(geodata['K_wind'][i]) > 0.1:
                    turbine_icon = folium.features.CustomIcon('%s/Icon/wind-power.png'%os.getcwd(), icon_size=icon_size)
                    folium.Marker(
                        location=[row['geometry'].y+delta_y, row['geometry'].x+delta_x],
                        icon=turbine_icon,
                        popup=row['#Name'],
                    ).add_to(m)
    
                    if notation == True:
                        character = row['#Name'] + ' ' + '%sMW'%(df['K_wind'][i])
                        folium.Marker(
                            location=[row['geometry'].y-0.03+delta_y, row['geometry'].x-0.03+delta_x],
                            icon=DivIcon(
                            html='<<div style="font-size: %spt"><b>%s</b></div>' % (fontsize,character),
                                ),
                            popup=row['#Name'],  # Use the "Name" column as the label
                        ).add_to(m)
                    
                elif float(geodata['K_PV'][i]) > 0.1 and float(geodata['K_wind'][i]) < 0.1:
                    
                    pv_icon = folium.features.CustomIcon('%s/Icon/solar-power.png'%os.getcwd(), icon_size=icon_size)
                    folium.Marker(
                        location=[row['geometry'].y+delta_y, row['geometry'].x+delta_x],
                        icon=pv_icon,
                        popup=row['#Name'],
                    ).add_to(m)
    
                    if notation == True:
                        character = row['#Name'] + ' ' + '%sMW'%(df['K_PV'][i])
                        folium.Marker(
                            location=[row['geometry'].y-0.03+delta_y, row['geometry'].x-0.03+delta_x],
                            icon=DivIcon(
                            html='<<div style="font-size: %spt"><b>%s</b></div>' % (fontsize,character),
                                ),
                            popup=row['#Name'],  # Use the "Name" column as the label
                        ).add_to(m)
            
            elif row['#Name'] == 'End user':
                #user_location = [row['geometry'].y, row['geometry'].x]
                user_icon = folium.features.CustomIcon('%s/Icon/factory.png'%os.getcwd(), icon_size=icon_size)
                folium.Marker(
                    location=[row['geometry'].y, row['geometry'].x+0.12],
                    icon=user_icon,
                    popup=row['#Name'],  # Use the "Name" column as the label
                ).add_to(m)
                
                
            elif row['#Name'] == 'Electrolyser':
                #el_location = [row['geometry'].y, row['geometry'].x+0.1]
                electro_icon = folium.features.CustomIcon('%s/Icon/electrolyser.png'%os.getcwd(), icon_size=icon_size)
                folium.Marker(
                    location=[row['geometry'].y, row['geometry'].x-0.13],
                    icon=electro_icon,
                    popup=row['#Name'],  # Use the "Name" column as the label
                ).add_to(m)
                if notation == True:
                    character = '%sMW'%(el_capacity) # + '%sMW'%(el_capacity)
                    folium.Marker(
                        location=[row['geometry'].y-0.03, row['geometry'].x-0.18],
                        icon=DivIcon(
                            html='<<div style="font-size: %spt"><b>%s</b></div>' % (fontsize,character),
                            ),
                        popup=row['#Name'],  # Use the "Name" column as the label
                    ).add_to(m)
                
            
            elif row['#Name'] == 'Storage':
                user_icon = folium.features.CustomIcon('%s/Icon/gas-storage.png'%os.getcwd(), icon_size=icon_size)
                folium.Marker(
                    location=[row['geometry'].y, row['geometry'].x+0.05],
                    icon=user_icon,
                    popup=row['#Name'],  # Use the "Name" column as the label
                ).add_to(m)
                if notation == True:
                    character = '%st'%(stg_capacity)
                    folium.Marker(
                        location=[row['geometry'].y-0.03, row['geometry'].x+0.03],
                        icon=DivIcon(
                            html='<<div style="font-size: %spt"><b>%s</b></div>' % (fontsize,character),
                            ),
                        popup=row['#Name'],  # Use the "Name" column as the label
                    ).add_to(m)
        
        # legend
        if notation == True:
            factor = 1.5
            user_icon = folium.features.CustomIcon('%s/Icon/north.png'%os.getcwd(), icon_size=icon_size)
            folium.Marker(
                location=[centre_lat+0.3*factor, centre_long+0.7*factor],
                icon=user_icon,
                popup=row['#Name'],  # Use the "Name" column as the label
            ).add_to(m)
            
            line = LineString([Point(centre_long+0.7*factor,centre_lat+0.25*factor), Point(centre_long+0.5*factor,centre_lat+0.25*factor)])
            line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
            line_style = {
            'color': 'red',  # Change the line color (e.g., to red)
            'weight': 3,  # Change the line width
            }
            folium.GeoJson(line_gdf, style_function=lambda x: line_style).add_to(m)
            
            character = 'Transmission'
            folium.Marker(
                location=[centre_lat+0.26*factor, centre_long+0.75*factor],
                icon=DivIcon(
                    html='<div style="font-size: 14pt"><b>%s</b></div>' % character,
                    ),
                popup=row['#Name'],  # Use the "Name" column as the label
            ).add_to(m)
            
            # for pipe line
            line = LineString([Point(centre_long+0.7*factor,centre_lat+0.2*factor), Point(centre_long+0.5*factor,centre_lat+0.2*factor)])
            line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
            line_style2 = {
            'color': 'black',  # Line color
            'weight': 3,  # Line width
            'dashArray': '5, 10',  # Length of dashes and gaps (5 pixels, 10 pixels)
            'dashOffset': '0',}
            folium.GeoJson(line_gdf, style_function=lambda x: line_style2).add_to(m)
            
            character = 'Pipeline'
            folium.Marker(
                location=[centre_lat+0.21*factor, centre_long+0.75*factor],
                icon=DivIcon(
                    html='<div style="font-size: 14pt"><b>%s</b></div>' % character,
                    ),
                popup=row['#Name'],  # Use the "Name" column as the label
            ).add_to(m)
        
        # Save the map
        import io
        from PIL import Image
        
        img_data = m._to_png(5)
        img = Image.open(io.BytesIO(img_data))
        img.save(os.getcwd()+'/image_%s.png'%Results['El'].values[k])

#plot_GIS_stg()
