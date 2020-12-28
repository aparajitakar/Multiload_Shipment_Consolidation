# import numpy as np
import pandas as pd
from ComputeDistance import ComputeDistance
from clustering import clustering
from heuristic import heuristic
pd.options.mode.chained_assignment = None  # default='warn'

# Import production data
file_pd = 'input/pd.csv'
df = pd.read_csv(file_pd)

# Drop columns
df = df[['FAMILY','ORDER','ORIG_PROM','CITY','STATE','LINE SHIP POSTAL CODE',
         'HEIGHT','LENGTH','WIDTH','WEIGHT']]

# Clean production zipcode
weight_object = df[df['LINE SHIP POSTAL CODE'].str.isdigit() == False].index
df.loc[weight_object, 'LINE SHIP POSTAL CODE'] = df.loc[weight_object]['LINE SHIP POSTAL CODE'].str.replace("-", " ")

# Add latitude and longitude
# Import zipcode file
zipcode = pd.read_csv('input/zipcode_all.csv')
zipcode['Zip Code'] = zipcode['Zip Code'].astype('str')
zipcode = zipcode.rename(columns={'Zip Code':'LINE SHIP POSTAL CODE'})

# Join production data and zipcode file
df = pd.merge(df, zipcode, how='left')
# df = df.drop_duplicates(subset=['ORDER'], keep='first')

# data cleaning
#remove rows having destinations "LAREDO" and "Mexico"

indexNames1 = df[df['CITY'] == "LAREDO"].index
df.drop(indexNames1 , inplace=True)

indexNames2 = df[df['CITY'] == "Mexico"].index
df.drop(indexNames2 , inplace=True)

# #adding latitudes and longitudes to quebec region
# q_lat = 45.5017
# q_long = 73.5673
# df.loc[df.CITY == "Montreal", 'Latitude'] = df.loc[df.CITY == "Montreal", 'Latitude'].fillna(q_lat)
# df.loc[df.CITY == "Montreal", 'Longitude'] = df.loc[df.CITY == "Montreal", 'Longitude'].fillna(q_long)

#replace "Weight" as float64 instead of object
#converting one weight data having comma
weight_object = df[df['WEIGHT'].str.isdigit() == False].index
df.loc[weight_object, 'WEIGHT'] = df.loc[weight_object]['WEIGHT'].str.replace(",","")
df['WEIGHT'] = df.WEIGHT.astype(float)

df.astype({'ORDER': 'object'})

#filling missing values by getting averages from Family of shipments

mean = pd.DataFrame(data = df.groupby('FAMILY').mean())
df_mean = mean[["HEIGHT","LENGTH","WIDTH","WEIGHT"]]

x = ["FAN_COILS","UNIT_VENT","WSHP"]
y = ["LENGTH","HEIGHT","WIDTH","WEIGHT"]

for i in x:
    for j in y:
        df.loc[df.FAMILY == i,j] = df.loc[df.FAMILY == i,j].fillna(df_mean.loc[i].loc[j])

# add distance and estimated time
orders = ComputeDistance(df)
df = pd.merge(df, orders,how='left')

# transform negative longitude
positive_long = df[df['Longitude'] > 0].index
df.loc[positive_long,'Longitude'] = df.loc[positive_long,'Longitude']* (-1)

# clustering
df = clustering(df)

# group by order and calculate the sum of weight and volumes
df['VOLUME'] = df['HEIGHT']*df['LENGTH']*df['WIDTH']
df['WEIGHT'] = pd.to_numeric(df['WEIGHT'],errors='coerce')
weights = df.groupby('ORDER')['WEIGHT'].sum().reset_index()
volumes = df.groupby('ORDER')['VOLUME'].sum().reset_index()
orders = weights.merge(volumes, on='ORDER')

# merge with df to get distance and cluster_label information
temp = df.drop_duplicates(subset=['ORDER'], keep='first')
orders = orders.merge(temp[['ORDER','DISTANCE','cluster_label']], on='ORDER', how='left')

# heuristic algorithm
output = heuristic(orders)

# output excel file
output['COST DIFF'] = output['LTL COST'] - output['TL COST']
output = output.sort_values(by='COST DIFF',ascending=False)
output['ZONE'] = output['ZONE'].astype(float)
output['ZONE'] = output['ZONE'].astype(int)
output.to_csv("output/output.csv", index=False)

print("Success! Please check the 'output.csv' file for results.")