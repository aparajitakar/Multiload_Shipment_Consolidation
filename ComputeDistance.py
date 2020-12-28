# installation needed: command line enter: $ pip install -U googlemaps
# import packages
import googlemaps
from datetime import datetime
import pandas as pd
from pandas.io.json import json_normalize
pd.options.mode.chained_assignment = None  # default='warn'

def ComputeDistance(df):
    df = df.drop_duplicates(subset=['ORDER'], keep='first')
    n_orders = df.shape[0]

    # enter your google API key
    # you need to get your own API key
    # See https://developers.google.com/maps/documentation/distance-matrix/get-api-key
    gmaps = googlemaps.Client(key='add api key here')

    # create origins and destinations
    origins = ["Laredo,TX"]
    destinations = []
    for i in df.index:
        destinations.append(str(str(df['CITY'][i]) + ' ' + str(df['STATE'][i])))

    # send request to distance matrix API (get the first 25 entries)
    matrix = gmaps.distance_matrix(origins, destinations[0:25])

    # normalize json response to dataframe
    dist = pd.json_normalize(matrix, ['rows', 'elements'])

    # normalize json response to dataframe
    # batch size = 25 because the usage limit is 25 destinations per request
    n_seg = n_orders // 25
    for i in range(1, n_seg):
        # send request to distance matrix API (except for the last segment)
        matrix = gmaps.distance_matrix(origins, destinations[0 + 25 * i:25 + 25 * i])
        temp = pd.json_normalize(matrix, ['rows', 'elements'])
        dist = dist.append(temp)

    # send request to distance matrix API (the last segment)
    matrix = gmaps.distance_matrix(origins, destinations[0 + 25 * n_seg:])
    temp = pd.json_normalize(matrix, ['rows', 'elements'])
    dist = dist.append(temp)

    # transform meters to mile
    mileage = dist['distance.value'] * 0.000621371192
    mileage = mileage.tolist()
    df['DISTANCE'] = mileage

    orders = df[['ORDER','DISTANCE']]

    return orders
