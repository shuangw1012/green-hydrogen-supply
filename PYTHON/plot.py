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

def plot_GIS():
    import geopandas as gpd
    from shapely.geometry import Point,Polygon,LineString
    import matplotlib.pyplot as plt
    import folium
    from folium.features import DivIcon
    # Load your data from the CSV file
    Results = pd.read_csv(os.path.join(os.getcwd(), 'results_2020.csv'))
    data = pd.read_csv(os.path.join(os.getcwd(), 'input_tas.txt'))
    
    for k in range(len(Results['El'].values)):
        Plot_results = np.array([])
        for j in range(len(data['#Name'])-1):
            if Results['wind_capacity_%s[kW]'%data['#Name'][j]][k]!=0:
                Plot_results=np.append(Plot_results,[data['#Name'][j],data['Lat'][j],data['Long'][j],
                                                     int(Results['wind_capacity_%s[kW]'%data['#Name'][j]][k]/1000),
                                                     int(data['Area'][j]*5.2)])
        Plot_results = Plot_results.reshape(int(len(Plot_results)/5),5)
        el_lat = data[data['#Name']==Results['El'][k]]['Lat'].iloc[0]
        el_long = data[data['#Name']==Results['El'][k]]['Long'].iloc[0]
        
        df = pd.DataFrame(Plot_results, columns=['#Name', 'Lat', 'Long','K_wind','K_wind_max'])
        
        df = df.append({'#Name': 'Electrolyser', 'Lat': el_lat, 'Long': el_long}, ignore_index=True)
        df = df.append({'#Name': 'End user', 'Lat': data['Lat'].values[-1], 'Long': data['Long'].values[-1]}, ignore_index=True)
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
        img.save('image_%s.png'%Results['El'].values[k])
        #m.save('map_%s.png'%Results['El'].values[i])

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
    m = folium.Map(location=[-40.95, 145.5], zoom_start=10.4)
    
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
    img.save('image.png')
    #m.save('map_%s.png'%Results['El'].values[i])

plot_GIS2()

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