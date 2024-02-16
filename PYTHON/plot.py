import matplotlib.pyplot as plt
import numpy as np

import os

plt.rcParams["font.family"] = "Times New Roman"
from projdirs import datadir
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
    Filter_location = ['Salt-100','Salt-90','Rock-100','Rock-90', 'Pipe-100', 'Pipe-90', 'Depleted-100', 'Depleted-90']
    #Filter_location = ['Salt Cavern','Lined Rock','Pipe Storage','Depleted Gas\nUSG','Depleted Gas\nMoomba']
    cost_categories = ['Wind', 'Electrolyser', 'Storage', 'Transmission', 'Transportation']
  
    data_multi = np.array([[1.64,1.30,0.10,0.00,0.08],
                           [1.65,1.31,0.03,0.00,0.09],
                           [1.64,1.30,0.18,0.00,0.04],
                           [1.65,1.31,0.05,0.00,0.04],
                           [1.90,1.51,0.68,0.01,0.04],
                           [1.69,1.34,0.20,0.00,0.04],
                           [1.90,1.50,0.05,0.03,0.07],
                           [1.90,1.50,0.02,0.03,0.07]])
    
    data = data_multi
    
    colors = ['lightblue', 'lightpink', 'gray', 'lightgreen', 'black']#, 'orange', 'blue', 'magenta', 'brown', 'pink']
    
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
    plt.ylim(0,5.2)
    
    for j in range(len(Filter_location)):
        bottom = Bottom[j]
        for i in range(len(cost_categories)):
            if data[j,i]==0:
                continue
            
            if i==2:
                plt.text(Filter_location[j], Bottom[j][i]-1.*data[j,i], str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
            elif i==3:
                plt.text(Filter_location[j], Bottom[j][i]+0.12, str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
            elif i==4:
                plt.text(Filter_location[j], Bottom[j][i]+0.32, str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
            else:
                plt.text(Filter_location[j], Bottom[j][i]-0.7*data[j,i], str((data[j,i])), ha='center', va='bottom', fontsize=14,weight='bold')
        #plt.text(Filter_location[j], bottom[i], str(round(sum(data[j,:]),2)), ha='center', va='bottom', fontsize=10,weight='bold')
    
    plt.tick_params(axis='both', labelsize=12)
    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.getcwd()+'/lcoh.png',dpi=100)
    plt.close(fig)

plot_bar2()

def plot_bar3():
    fig = plt.figure(figsize=(3, 5))
    cost_categories = ['Wind', 'Electrolyser', 'Storage', 'Transmission', 'Transportation']
    colors = ['blue', 'lightpink', 'gray', 'lightgreen', 'black']#, 'orange', 'blue', 'magenta', 'brown', 'pink']
    values = np.array([1.42,1.13,0.12,0.28,0.00])
    values = np.array([1.55,1.23,0.14,0.03,0.00])
    values = np.array([1.42,1.13,0.12,0.00,0.03])
    values = np.array([1.55,1.23,0.14,0.00,0.015])
    plt.bar(cost_categories, values, color=colors,width = 0.5)
    plt.xticks(rotation=45)
    for i in range(len(cost_categories)):
        plt.text(cost_categories[i], values[i]+0.02, str(round(values[i],3)), ha='center', va='bottom', fontsize=10,weight='bold')
    plt.ylabel('LCOH (USD/kg)',fontsize = 14)
    plt.ylim(0,1.7)
    plt.tight_layout()
    plt.savefig(os.getcwd()+'/lcoh-breakdown-4.png',dpi=400)
    plt.close(fig)



def plot_bar4():
    fig = plt.figure(figsize=(7, 5))
    cost_categories = ['Wind', 'Electrolyser', 'Storage', 'Transmission', 'Transportation']
    colors = ['blue', 'pink', 'gray', 'green', 'black']#, 'orange', 'blue', 'magenta', 'brown', 'pink']
    Index = np.linspace(1,len(cost_categories),len(cost_categories))
    values3 = np.array([1.45,1.15,0.19,0.07,0.00])
    values2 = np.array([1.42,1.13,0.26,0.11,0.01])
    values1 = np.array([1.44,1.14,0.14,0.02,0.03])
    plt.bar(Index-0.3, values1, color=colors,width = 0.2,label = 'Electrolyser at end user',edgecolor='black',linewidth = 1.5)
    plt.bar(Index, values2, color=colors,width = 0.2,label = 'Electrolyser along pipeline',edgecolor='yellow',linewidth = 1.5)
    plt.bar(Index+0.3, values3, color=colors,width = 0.2,label = 'Electrolyser at best resource',edgecolor='red',linewidth = 1.5)
    plt.xticks(rotation=45)
    plt.xticks(Index,cost_categories,rotation=45)
    for i in range(len(cost_categories)):
        plt.text(Index[i]-0.3, values1[i]+0.03, str(round(values1[i],3)), ha='center', va='bottom', fontsize=10,weight='bold')
        plt.text(Index[i], values2[i]+0.03, str(round(values2[i],3)), ha='center', va='bottom', fontsize=10,weight='bold')
        plt.text(Index[i]+0.3, values3[i]+0.03, str(round(values3[i],3)), ha='center', va='bottom', fontsize=10,weight='bold')
    
    
    plt.legend()
    plt.ylabel('LCOH (USD/kg)',fontsize = 14)
    plt.ylim(0,1.7)
    plt.tight_layout()
    plt.savefig(os.getcwd()+'/lcoh-breakdown-5.png',dpi=400)
    plt.close(fig)


def plot_GIS():
    import geopandas as gpd
    from shapely.geometry import Point,Polygon,LineString
    import matplotlib.pyplot as plt
    import folium
    from folium.features import DivIcon
    # Load your data from the CSV file
    Results = pd.read_csv(os.path.join(os.getcwd(), 'results_2020_30MW.csv'))
    data = pd.read_csv(os.path.join(os.getcwd(), 'input_usg.txt'))
    
    for k in range(len(Results['El'].values)):
        #if data['#Name'][k]!= 'KF249':
        #    continue
        Plot_results = np.array([])
        for j in range(len(data['#Name'])-1):
            try:
                if Results['wind_capacity_%s[kW]'%data['#Name'][j]][k]>5000:
                    Plot_results=np.append(Plot_results,[data['#Name'][j],data['Lat'][j],data['Long'][j],
                                                         int(Results['wind_capacity_%s[kW]'%data['#Name'][j]][k]/1000),
                                                         int(data['Area'][j]*5.2)])
            except:
                #if data['#Name'][j]!= 'KF249':
                #    continue
                Plot_results=np.append(Plot_results,[data['#Name'][j],data['Lat'][j],data['Long'][j],
                                                     int(Results['wind_capacity[kW]'][k]/1000),
                                                     int(data['Area'][j]*5.2)])
        Plot_results = Plot_results.reshape(int(len(Plot_results)/5),5)
        el_lat = data[data['#Name']==Results['El'][k]]['Lat'].iloc[0]
        el_long = data[data['#Name']==Results['El'][k]]['Long'].iloc[0]
        
        df = pd.DataFrame(Plot_results, columns=['#Name', 'Lat', 'Long','K_wind','K_wind_max'])
        
        df = df.append({'#Name': 'Electrolyser', 'Lat': el_lat, 'Long': el_long}, ignore_index=True)
        df = df.append({'#Name': 'End user', 'Lat': data['Lat'].values[-1], 'Long': data['Long'].values[-1]}, ignore_index=True)
        #df = df.append({'#Name': 'Storage', 'Lat': data['Lat'].values[-2], 'Long': data['Long'].values[-2]}, ignore_index=True)
        
        crs = {'init': 'epsg:4326'}
        geometry = [Point(xy) for xy in zip(df["Long"], df["Lat"])]
        geodata = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
        # Create a Folium map
        m = folium.Map(location=[-32.8, 137.35], zoom_start=10.4)
        #m = folium.Map(location=[-40.95, 145.3], zoom_start=10.45)
        # Add the connection line (if needed)
        for i in range(len(df)-2):
            line = LineString([geodata.iloc[i]['geometry'], geodata.iloc[-2]['geometry']])
            line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
            #folium.GeoJson(line_gdf).add_to(m)
            line_style = {
            'color': 'red',  # Change the line color (e.g., to red)
            'weight': 3,  # Change the line width
            }
            folium.GeoJson(line_gdf, style_function=lambda x: line_style).add_to(m)
            
        # for pipe line
        line = LineString([geodata.iloc[-2]['geometry'], geodata.iloc[-1]['geometry']])
        line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
        line_style2 = {
        'color': 'black',  # Line color
        'weight': 3,  # Line width
        'dashArray': '5, 10',  # Length of dashes and gaps (5 pixels, 10 pixels)
        'dashOffset': '0',}
        folium.GeoJson(line_gdf, style_function=lambda x: line_style2).add_to(m)

        # Add your original GeoDataFrame (points) to the map with labels
        for i, row in geodata.iterrows():
            if row['#Name'] != 'Electrolyser' and row['#Name'] != 'End user':
                if row['geometry'].y == el_lat and row['geometry'].x == el_long:
                    
                    turbine_icon = folium.features.CustomIcon('%s/Icon/wind-power.png'%os.getcwd(), icon_size=(50,50))
                    folium.Marker(
                        location=[row['geometry'].y, row['geometry'].x+0.04],
                        icon=turbine_icon,
                        popup=row['#Name'],
                    ).add_to(m)
    
                    
                    character = row['#Name'] + ' ' + '%sMW'%(df['K_wind'][i])
                    folium.Marker(
                        location=[row['geometry'].y-0.03, row['geometry'].x+0.01],
                        icon=DivIcon(
                            html='<<div style="font-size: 14pt"><b>%s</b></div>' % character,
                            ),
                        popup=row['#Name'],  # Use the "Name" column as the label
                    ).add_to(m)
                    
                else:
                    turbine_icon = folium.features.CustomIcon('%s/Icon/wind-power.png'%os.getcwd(), icon_size=(50,50))
                    folium.Marker(
                        location=[row['geometry'].y, row['geometry'].x],
                        icon=turbine_icon,
                        popup=row['#Name'],
                    ).add_to(m)
    
                    
                    character = row['#Name'] + ' ' + '%sMW'%(df['K_wind'][i])
                    folium.Marker(
                        location=[row['geometry'].y-0.03, row['geometry'].x-0.03],
                        icon=DivIcon(
                            html='<<div style="font-size: 14pt"><b>%s</b></div>' % character,
                            ),
                        popup=row['#Name'],  # Use the "Name" column as the label
                    ).add_to(m)
            
            elif row['#Name'] == 'End user':
                user_icon = folium.features.CustomIcon('%s/Icon/factory.png'%os.getcwd(), icon_size=(50,50))
                folium.Marker(
                    location=[row['geometry'].y, row['geometry'].x+0.05],
                    icon=user_icon,
                    popup=row['#Name'],  # Use the "Name" column as the label
                ).add_to(m)

            elif row['#Name'] == 'Electrolyser':
                electro_icon = folium.features.CustomIcon('%s/Icon/electrolyser.png'%os.getcwd(), icon_size=(50,50))
                folium.Marker(
                    location=[row['geometry'].y, row['geometry'].x],
                    icon=electro_icon,
                    popup=row['#Name'],  # Use the "Name" column as the label
                ).add_to(m)
                
            
        
        # Save the map
        import io
        from PIL import Image
        
        img_data = m._to_png(5)
        img = Image.open(io.BytesIO(img_data))
        img.save(os.getcwd()+'/image_%s.png'%Results['El'].values[k])

#plot_GIS()

def plot_GIS_storage():
    import geopandas as gpd
    from shapely.geometry import Point,Polygon,LineString
    import matplotlib.pyplot as plt
    import folium
    from folium.features import DivIcon
    # Load your data from the CSV file
    Results = pd.read_csv(os.path.join(os.getcwd(), 'results_2020_3000MW.csv'))
    data = pd.read_csv(os.path.join(os.getcwd(), 'input_usg_plot.txt'))
    
    for k in range(len(Results['El'].values)):
        #if data['#Name'][k]!= 'KF249':
        #    continue
        Plot_results = np.array([])
        for j in range(len(data['#Name'])-2):
            try:
                if Results['wind_capacity_%s[kW]'%data['#Name'][j]][k]>5000:
                    Plot_results=np.append(Plot_results,[data['#Name'][j],data['Lat'][j],data['Long'][j],
                                                         int(Results['wind_capacity_%s[kW]'%data['#Name'][j]][k]/1000),
                                                         int(data['Area'][j]*5.2)])
            except:
                #if data['#Name'][j]!= 'KF249':
                #    continue
                Plot_results=np.append(Plot_results,[data['#Name'][j],data['Lat'][j],data['Long'][j],
                                                     int(Results['wind_capacity[kW]'][k]/1000),
                                                     int(data['Area'][j]*5.2)])
                
        Plot_results = Plot_results.reshape(int(len(Plot_results)/5),5)
        el_lat = data[data['#Name']==Results['El'][k]]['Lat'].iloc[0]
        el_long = data[data['#Name']==Results['El'][k]]['Long'].iloc[0]
        
        df = pd.DataFrame(Plot_results, columns=['#Name', 'Lat', 'Long','K_wind','K_wind_max'])
        
        df = df.append({'#Name': 'Electrolyser', 'Lat': el_lat, 'Long': el_long}, ignore_index=True)
        df = df.append({'#Name': 'End user', 'Lat': data['Lat'].values[-1], 'Long': data['Long'].values[-1]}, ignore_index=True)
        df = df.append({'#Name': 'Storage', 'Lat': data['Lat'].values[-2], 'Long': data['Long'].values[-2]}, ignore_index=True)
        
        crs = {'init': 'epsg:4326'}
        geometry = [Point(xy) for xy in zip(df["Long"], df["Lat"])]
        geodata = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
        # Create a Folium map
        m = folium.Map(location=[-32.8, 137.35], zoom_start=10.4)
        #m = folium.Map(location=[-28.11,140.23], zoom_start=10.4)
        #m = folium.Map(location=[-30, 139], zoom_start=7)
        # Add the connection line (if needed)
        for i in range(len(df)-2):
            #continue
            line = LineString([geodata.iloc[i]['geometry'], geodata.iloc[-3]['geometry']])
            line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
            #folium.GeoJson(line_gdf).add_to(m)
            line_style = {
            'color': 'red',  # Change the line color (e.g., to red)
            'weight': 3,  # Change the line width
            }
            folium.GeoJson(line_gdf, style_function=lambda x: line_style).add_to(m)
            
        # for pipe line
        line = LineString([geodata.iloc[-2]['geometry'], geodata.iloc[-1]['geometry']])
        line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
        line_style2 = {
        'color': 'black',  # Line color
        'weight': 3,  # Line width
        'dashArray': '5, 10',  # Length of dashes and gaps (5 pixels, 10 pixels)
        'dashOffset': '0',}
        folium.GeoJson(line_gdf, style_function=lambda x: line_style2).add_to(m)
        
        line = LineString([geodata.iloc[-3]['geometry'], geodata.iloc[-1]['geometry']])
        line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
        line_style2 = {
        'color': 'black',  # Line color
        'weight': 3,  # Line width
        'dashArray': '5, 10',  # Length of dashes and gaps (5 pixels, 10 pixels)
        'dashOffset': '0',}
        folium.GeoJson(line_gdf, style_function=lambda x: line_style2).add_to(m)

        # Add your original GeoDataFrame (points) to the map with labels
        for i, row in geodata.iterrows():
            if row['#Name'] != 'Electrolyser' and row['#Name'] != 'End user' and row['#Name'] != 'Storage':
                if row['geometry'].y == el_lat and row['geometry'].x == el_long:
                    #continue
                    turbine_icon = folium.features.CustomIcon('%s/Icon/wind-power.png'%os.getcwd(), icon_size=(50,50))
                    folium.Marker(
                        location=[row['geometry'].y, row['geometry'].x+0.04],
                        icon=turbine_icon,
                        popup=row['#Name'],
                    ).add_to(m)
    
                    
                    character = row['#Name'] + ' ' + '%sMW'%(df['K_wind'][i])
                    folium.Marker(
                        location=[row['geometry'].y-0.03, row['geometry'].x+0.01],
                        icon=DivIcon(
                            html='<div style="font-size: 12pt">%s</div>' % character,
                            ),
                        popup=row['#Name'],  # Use the "Name" column as the label
                    ).add_to(m)
                    
                else:
                    #continue
                    turbine_icon = folium.features.CustomIcon('%s/Icon/wind-power.png'%os.getcwd(), icon_size=(50,50))
                    folium.Marker(
                        location=[row['geometry'].y, row['geometry'].x],
                        icon=turbine_icon,
                        popup=row['#Name'],
                    ).add_to(m)
    
                    
                    character = row['#Name'] + ' ' + '%sMW'%(df['K_wind'][i])
                    folium.Marker(
                        location=[row['geometry'].y-0.03, row['geometry'].x-0.03],
                        icon=DivIcon(
                            html='<div style="font-size: 12pt">%s</div>' % character,
                            ),
                        popup=row['#Name'],  # Use the "Name" column as the label
                    ).add_to(m)
                

            elif row['#Name'] == 'Electrolyser':
                electro_icon = folium.features.CustomIcon('%s/Icon/electrolyser.png'%os.getcwd(), icon_size=(50,50))
                folium.Marker(
                    location=[row['geometry'].y, row['geometry'].x+0.1],
                    icon=electro_icon,
                    popup=row['#Name'],  # Use the "Name" column as the label
                ).add_to(m)
                
            elif row['#Name'] == 'End user':
                user_icon = folium.features.CustomIcon('%s/Icon/factory.png'%os.getcwd(), icon_size=(50,50))
                folium.Marker(
                    location=[row['geometry'].y, row['geometry'].x+0.05],
                    icon=user_icon,
                    popup=row['#Name'],  # Use the "Name" column as the label
                ).add_to(m)
                
            elif row['#Name'] == 'Storage':
                user_icon = folium.features.CustomIcon('%s/Icon/gas-storage.png'%os.getcwd(), icon_size=(50,50))
                folium.Marker(
                    location=[row['geometry'].y, row['geometry'].x-0.05],
                    icon=user_icon,
                    popup=row['#Name'],  # Use the "Name" column as the label
                ).add_to(m)
        
        # Save the map
        import io
        from PIL import Image
        
        img_data = m._to_png(5)
        img = Image.open(io.BytesIO(img_data))
        img.save(os.getcwd()+'/image_%s.png'%Results['El'].values[k]) 

#plot_GIS_storage()
        
def plot_GIS2():
    import geopandas as gpd
    from shapely.geometry import Point,Polygon,LineString
    import matplotlib.pyplot as plt
    import folium
    from folium.features import DivIcon
    # Load your data from the CSV file
    data = pd.read_csv(os.path.join(os.getcwd(), 'input_tas.txt'))
    
    crs = {'init': 'epsg:4326'}
    geometry = [Point(xy) for xy in zip(data["Long"], data["Lat"])]
    geodata = gpd.GeoDataFrame(data, crs=crs, geometry=geometry)
    # Create a Folium map
    m = folium.Map(location=[-41, 145.5], zoom_start=10.4)
    
    for i, row in geodata.iterrows():
        if row['#Name'] != 'User':
            turbine_icon = folium.features.CustomIcon('%s/Icon/wind-power.png'%os.getcwd(), icon_size=(50,50))
            folium.Marker(
                location=[row['geometry'].y, row['geometry'].x],
                icon=turbine_icon,
                popup=row['#Name'],
            ).add_to(m)

            character = row['#Name'] + ' ' + '%sMW'%(int(data['Area'][i]*5.2))
            folium.Marker(
                location=[row['geometry'].y-0.03, row['geometry'].x-0.03],
                icon=DivIcon(
                    html='<div style="font-size: 12pt">%s</div>' % character,
                    ),
                popup=row['#Name'],  # Use the "Name" column as the label
            ).add_to(m)
            
        elif row['#Name'] == 'User':
            user_icon = folium.features.CustomIcon('%s/Icon/factory.png'%os.getcwd(), icon_size=(50,50))
            folium.Marker(
                location=[row['geometry'].y, row['geometry'].x+0.05],
                icon=user_icon,
                popup=row['#Name'],  # Use the "Name" column as the label
            ).add_to(m)
    
    # Save the map
    import io
    from PIL import Image
    
    img_data = m._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    img.save(os.getcwd()+'/image.png')
    #m.save('map_%s.png'%Results['El'].values[i])

def plot_GIS3():
    import geopandas as gpd
    from shapely.geometry import Point,Polygon,LineString
    import matplotlib.pyplot as plt
    import folium
    from folium.features import DivIcon
    # Load your data from the CSV file
    Results = pd.read_csv(os.path.join(os.getcwd(), 'results_2020_user.csv'))
    data = pd.read_csv(os.path.join(os.getcwd(), 'input_tas.txt'))
    
    for k in range(1):#len(Results['El'].values)):
        Plot_results = np.array([])
        for j in range(len(data['#Name'])-1):
            if Results['wind_capacity_%s[kW]'%data['#Name'][j]][k]>5000:
                Plot_results=np.append(Plot_results,[data['#Name'][j],data['Lat'][j],data['Long'][j],
                                                     int(Results['wind_capacity_%s[kW]'%data['#Name'][j]][k]/1000),
                                                     int(data['Area'][j]*5.2)])
        Plot_results = Plot_results.reshape(int(len(Plot_results)/5),5)
        el_lat = data[data['#Name']==Results['El'][k]]['Lat'].iloc[0]
        el_long = data[data['#Name']==Results['El'][k]]['Long'].iloc[0]
        
        df = pd.DataFrame(Plot_results, columns=['#Name', 'Lat', 'Long','K_wind','K_wind_max'])
        
        df = df.append({'#Name': 'Electrolyser', 'Lat': el_lat, 'Long': el_long}, ignore_index=True)
        df = df.append({'#Name': 'End user', 'Lat': data['Lat'].values[-1], 'Long': data['Long'].values[-1]}, ignore_index=True)
        df = df.append({'#Name': 'Middle', 'Lat': -40.79, 'Long': 144.96}, ignore_index=True)
        crs = {'init': 'epsg:4326'}
        geometry = [Point(xy) for xy in zip(df["Long"], df["Lat"])]
        geodata = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
        # Create a Folium map
        m = folium.Map(location=[-40.95, 145.5], zoom_start=10.4)
        
        # Add the connection line (if needed)
        for i in range(len(df)-2):
            line = LineString([geodata.iloc[i]['geometry'], geodata.iloc[-2]['geometry']])
            line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
            #folium.GeoJson(line_gdf).add_to(m)
            line_style = {
            'color': 'red',  # Change the line color (e.g., to red)
            'weight': 3,  # Change the line width
            }
            folium.GeoJson(line_gdf, style_function=lambda x: line_style).add_to(m)
            
        # for second line
        line = LineString([geodata.iloc[0]['geometry'], geodata.iloc[-1]['geometry']])
        line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
        line_style2 = {
        'color': 'Red',  # Line color
        'weight': 3,  # Line width
        'dashArray': '5, 10',  # Length of dashes and gaps (5 pixels, 10 pixels)
        'dashOffset': '0',}
        folium.GeoJson(line_gdf, style_function=lambda x: line_style2).add_to(m)
        
        line = LineString([geodata.iloc[-2]['geometry'], geodata.iloc[-1]['geometry']])
        line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
        line_style2 = {
        'color': 'Red',  # Line color
        'weight': 3,  # Line width
        'dashArray': '5, 10',  # Length of dashes and gaps (5 pixels, 10 pixels)
        'dashOffset': '0',}
        folium.GeoJson(line_gdf, style_function=lambda x: line_style2).add_to(m)
        
        
        # Add your original GeoDataFrame (points) to the map with labels
        for i, row in geodata.iterrows():
            if row['#Name'] != 'Electrolyser' and row['#Name'] != 'End user' and row['#Name'] != 'Middle':
                if row['geometry'].y == el_lat and row['geometry'].x == el_long:
                    
                    turbine_icon = folium.features.CustomIcon('%s/Icon/wind-power.png'%os.getcwd(), icon_size=(50,50))
                    folium.Marker(
                        location=[row['geometry'].y, row['geometry'].x+0.04],
                        icon=turbine_icon,
                        popup=row['#Name'],
                    ).add_to(m)
    
                    
                    character = row['#Name'] + ' ' + '%sMW'%(df['K_wind'][i])
                    folium.Marker(
                        location=[row['geometry'].y-0.03, row['geometry'].x+0.01],
                        icon=DivIcon(
                            html='<div style="font-size: 12pt">%s</div>' % character,
                            ),
                        popup=row['#Name'],  # Use the "Name" column as the label
                    ).add_to(m)
                    
                else:
                    turbine_icon = folium.features.CustomIcon('%s/Icon/wind-power.png'%os.getcwd(), icon_size=(50,50))
                    folium.Marker(
                        location=[row['geometry'].y, row['geometry'].x],
                        icon=turbine_icon,
                        popup=row['#Name'],
                    ).add_to(m)
    
                    
                    character = row['#Name'] + ' ' + '%sMW'%(df['K_wind'][i])
                    folium.Marker(
                        location=[row['geometry'].y-0.03, row['geometry'].x-0.03],
                        icon=DivIcon(
                            html='<div style="font-size: 12pt">%s</div>' % character,
                            ),
                        popup=row['#Name'],  # Use the "Name" column as the label
                    ).add_to(m)
                

            elif row['#Name'] == 'Electrolyser':
                electro_icon = folium.features.CustomIcon('%s/Icon/electrolyser.png'%os.getcwd(), icon_size=(50,50))
                folium.Marker(
                    location=[row['geometry'].y, row['geometry'].x],
                    icon=electro_icon,
                    popup=row['#Name'],  # Use the "Name" column as the label
                ).add_to(m)
                
            elif row['#Name'] == 'End user':
                user_icon = folium.features.CustomIcon('%s/Icon/factory.png'%os.getcwd(), icon_size=(50,50))
                folium.Marker(
                    location=[row['geometry'].y, row['geometry'].x+0.05],
                    icon=user_icon,
                    popup=row['#Name'],  # Use the "Name" column as the label
                ).add_to(m)
        
        # Save the map
        import io
        from PIL import Image
        
        img_data = m._to_png(5)
        img = Image.open(io.BytesIO(img_data))
        img.save(os.getcwd()+'/image_%s.png'%Results['El'].values[k])
        
        
#plot_GIS2()
fontsize = 14

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

def plot_yearly_solar():
    title = np.array(['Burnie 1', 'Burnie 2', 'Burnie 3', 'Burnie 4', 'Gladstone 1', 'Gladstone 2', 'Gladstone 3', 
                      'Pilbara 1', 'Pilbara 2', 'Pilbara 3', 'Pilbara 4', 'Pinjarra 1', 'Pinjarra 2', 'Pinjarra 3', 'Pinjarra 4',
                      'USG 1', 'USG 2', 'USG 3', 'USG 4'])
    Data = np.array([[0.1420, 0.1630, 0.1681, 0.1536],
                    [0.1420, 0.1649, 0.1754, 0.1528],
                    [0.1500, 0.1653, 0.1729, 0.1565],
                    [0.1495, 0.1722, 0.1806, 0.1609],
                    [0.1800, 0.2263, 0.2159, 0.1902],
                    [0.1820, 0.2255, 0.2190, 0.1921],
                    [0.1850, 0.2232, 0.2327, 0.2019],
                    [0.2010, 0.2212, 0.2134, 0.2200],
                    [0.2150, 0.2130, 0.2181, 0.2186],
                    [0.2152, 0.2151, 0.2124, 0.2182],
                    [0.2160, 0.2098, 0.2121, 0.2204],
                    [0.1850, 0.1867, 0.1768, 0.1939],
                    [0.1870, 0.1853, 0.1791, 0.1975],
                    [0.1850, 0.1867, 0.1819, 0.1973],
                    [0.1845, 0.1834, 0.1728, 0.1927],
                    [0.1855, 0.2200, 0.2209, 0.1979],
                    [0.1750, 0.2156, 0.2161, 0.1886],
                    [0.1800, 0.2249, 0.2247, 0.2010],
                    [0.1820, 0.2212, 0.2232, 0.1978]])
    
    Index = np.linspace(1,len(title),len(title))
    plt.figure(figsize=(16, 8))
    #plt.bar(Index-0.3, Data[:,3], width=0.15, color='b', edgecolor='black', label='Wlab')
    plt.bar(Index-0.15, Data[:,2], width=0.15, color='green', edgecolor='black', label='Himawari')
    plt.bar(Index, Data[:,1], width=0.15, color='r', edgecolor='black', label='MERRA2')
    plt.bar(Index+0.15, Data[:,0], width=0.15, color='black', edgecolor='black', label='SolCast')
    plt.bar(Index+0.3, Data[:,3], width=0.15, color='pink', edgecolor='black', label='Atlas')
    
    #plt.xlabel(title,fontsize=fontsize,rotation=45)
    plt.ylabel('Capacity Factor',fontsize=fontsize)
    plt.xticks(Index-0.5,title,fontsize=fontsize,rotation=45)
    plt.yticks(fontsize=fontsize)
    plt.legend(ncol=5,fontsize=fontsize)
    plt.ylim(0,0.3)
    plt.savefig('%s/comparison_yearly_solar.png'%(os.getcwd()), dpi=500, bbox_inches='tight')

#plot_yearly_solar()

'''

# Display the numbers on top of the bars with adjusted positions
for case in cases:
    bottom = Bottom[cases.index(case)]
    for i, category in enumerate(cost_categories):
        value = cost_values[case][i]
        if value > 0:
            print (case,i,value,bottom[i])
            if i==3:
                plt.text(case, bottom[i]-2*value, str(value), ha='center', va='bottom', fontsize=12)
            elif case == 'Burnie 4' and i==4:
                plt.text(case, bottom[i]+0.5*value, str(value), ha='center', va='bottom', fontsize=12)
            elif case == 'Burnie 5' and i==4:
                print (bottom[i]+5*value)
                plt.text(case, bottom[i]+0.5*value, str(value), ha='center', va='bottom', fontsize=12)
            else:
                plt.text(case, bottom[i]-0.5*value, str(value), ha='center', va='bottom', fontsize=12)

# Show the plot
plt.tight_layout()
plt.savefig(os.getcwd()+'/Cost_breakdown.png',dpi=100)
plt.close(fig)
'''

'''
import matplotlib.pyplot as plt

# Sample cost breakdown data
cost_breakdown = {
    'PV': 0,
    'Wind': 909,
    'Electrolyser': 580,
    'UG storage': 12,
    'Transmission': 25,
    'FOM': 42
}

# Extract cost labels and values
cost_labels = list(cost_breakdown.keys())
cost_values = list(cost_breakdown.values())

# Create a histogram plot
fig = plt.figure(figsize=(6, 4))
colors = ['black', 'red', 'lightgreen', 'lightsalmon', 'lightpink']
plt.bar(cost_labels, cost_values, color=colors)
plt.ylabel('Cost (MUSD)',fontsize = 12)

# Rotate the x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Display the cost values on top of the bars (optional)
for i, value in enumerate(cost_values):
    plt.text(i, value, str(value), ha='center', va='bottom')

# Show the plot
plt.tight_layout()
plt.savefig(os.getcwd()+'/Best.png',dpi=100)
plt.close(fig)
'''