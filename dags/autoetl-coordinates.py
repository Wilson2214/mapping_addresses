
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable

import pandas as pd
from datetime import datetime
import geopandas as gpd
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import requests
import os
import gzip
import shutil
import gdown
from math import radians, cos, sin, asin, sqrt
import pickle 
from keplergl import KeplerGl

def get_input_data():

    print("Get Locations URL and Load into Pandas Dataframe")
    locations_url = "https://drive.google.com/file/d/1WOmj8wSpe8FDn_7opryh9tsng8ZqhAr1/view?usp=share_link"
    reconstructed_locations_url = 'https://drive.google.com/uc?id=' + locations_url.split('/')[-2]
    locations = pd.read_csv(reconstructed_locations_url)
    locations.to_csv('./data/inputs/locations.csv', index=False,)

    print("Get buildings data")
    buildings_url = "https://drive.google.com/file/d/1qO86txHm82OqWbEEIEsaevF_tRFv4PhK/view?usp=share_link"
    reconstructed_buildings_url = 'https://drive.google.com/uc?id=' + buildings_url.split('/')[-2]
    resp = urlopen(reconstructed_buildings_url)
    myzip = ZipFile(BytesIO(resp.read()))
    myzip.extractall('./data/inputs/')

    print("Get parcels data")
    parcels_url = "https://drive.google.com/file/d/137bCzf0qEPLwKTgNxeAAjTs5MvyTlxWA/view?usp=share_link"
    reconstructed_parcels_url = 'https://drive.google.com/uc?id=' + parcels_url.split('/')[-2]
    output = './data/inputs/ms_hinds_parcels.ndgeojson.gz'
    gdown.download(reconstructed_parcels_url, output, quiet=False)

    print("Extract and save gzip file")
    gzip_filename = './data/inputs/ms_hinds_parcels.ndgeojson.gz'
    dest_filepath = './data/inputs/ms_hinds_parcels.ndgeojson'
    with gzip.open(gzip_filename, 'rb') as file:
            with open(dest_filepath, 'wb') as output_file:
                output_file.write(file.read())

def get_new_geo():

    print("Load Data")
    locations = pd.read_csv('./data/inputs/locations.csv')
    buildings = gpd.read_file('./data/inputs/ms_hinds_buildings.json')
    buildings_join_tbl = pd.read_csv('./data/inputs/ms_hinds_buildings_join_table.csv')
    parcels = gpd.read_file('./data/inputs/ms_hinds_parcels.ndgeojson')

    print("Create Join Table")
    # Isolate from parcel: ll_uuid, address, szip
    parcels_join = parcels[['ll_uuid', 'address', 'szip']]
    # Confirm szip is 5 digit zip code to match f_ziplock
    parcels_join['szip'] = parcels_join['szip'].str.replace("-.*","")
    # Join to get ed_str_uuid and ed_bld_uuid
    buildings_join_tbl = buildings_join_tbl.drop('geoid', axis=1)
    parcels_join = parcels_join.merge(buildings_join_tbl, on='ll_uuid', how='left')
    # Drop all ll_uuid without a str and bld uuid
    parcels_join = parcels_join.dropna(subset=['ed_str_uuid','ed_bld_uuid'])
    # Join to buildings on both uuids to get ed_lat/ed_lon
    buildings_join = buildings[['ed_str_uuid', 'ed_bld_uuid', 'ed_lat', 'ed_lon']]
    parcels_join = parcels_join.merge(buildings_join, on=['ed_str_uuid', 'ed_bld_uuid'], how='left')
    buildings = buildings[['ed_str_uuid', 'ed_bld_uuid', 'ed_lat', 'ed_lon']]

    print("Define Functions")
    def hav_dist(lat1, long1, lat2, long2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
        # haversine formula 
        dlon = long2 - long1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        # Radius of earth in kilometers is 6371
        km = 6371* c
        return km

    def find_nearest(lat, lon, parcels_join_filt):
        distances = parcels_join_filt.apply(
            lambda row: hav_dist(lat, lon, row['ed_lat'], row['ed_lon']), 
            axis=1)
        return (parcels_join_filt.loc[distances.idxmin()]['ed_lat'], 
                parcels_join_filt.loc[distances.idxmin()]['ed_lon'],
                distances.min())

    def get_updated_geo(ll_uuid,lat,lon):
        # parcels_join is loaded previously
        # Filter parcels_join for all buildings associated with parsel
        parcels_join_filt = parcels_join[parcels_join['ll_uuid'] == ll_uuid]
        
        if parcels_join_filt.shape[0] == 0:
            # Calculate closest building
            if np.isnan(lat):
                return(None)
            else:
                # Filter buildings lat long for different ranges to make for a smaller subset
                close_buildings = buildings[(buildings['ed_lat'] >= lat - 0.001) & 
                                            (buildings['ed_lat'] <= lat + 0.001) & 
                                            (buildings['ed_lon'] >= lon - 0.001) & 
                                            (buildings['ed_lon'] <= lon + 0.001)]
                if close_buildings.shape[0] == 0:
                    # Refilter for a larger subset
                    close_buildings = buildings[(buildings['ed_lat'] >= lat - 0.01) & 
                                                (buildings['ed_lat'] <= lat + 0.01) & 
                                                (buildings['ed_lon'] >= lon - 0.01) & 
                                                (buildings['ed_lon'] <= lon + 0.01)]
                if close_buildings.shape[0] == 0:
                    # Refilter for a larger subset
                    close_buildings = buildings[(buildings['ed_lat'] >= lat - 0.1) & 
                                                (buildings['ed_lat'] <= lat + 0.1) & 
                                                (buildings['ed_lon'] >= lon - 0.1) & 
                                                (buildings['ed_lon'] <= lon + 0.1)]
                if close_buildings.shape[0] == 0:
                    # Refilter for a larger subset
                    close_buildings = buildings[(buildings['ed_lat'] >= lat - 1.0) & 
                                                (buildings['ed_lat'] <= lat + 1.0) & 
                                                (buildings['ed_lon'] >= lon - 1.0) & 
                                                (buildings['ed_lon'] <= lon + 1.0)]
                if close_buildings.shape[0] == 0:
                    return(None)
                else:
                    return(find_nearest(lat,lon,close_buildings))
            
        elif parcels_join_filt.shape[0] == 1:
            # There is only one building associated with this parsel
            # Return lat/long of this building as the updated coordinates
            if np.isnan(lat):
                distance = 0
            else:
                distance = hav_dist(lat, lon, parcels_join_filt.ed_lat, parcels_join_filt.ed_lon)
            return(parcels_join_filt.ed_lat.iloc[0], parcels_join_filt.ed_lon.iloc[0], distance)
        
        else:
            # There are multiple buildings associated with this matching parsel
            # Calculate the haversine distance between the location and each building
            # Return the closest building
            if np.isnan(lat):
                parcels_join_filt = parcels_join_filt.groupby("ll_uuid").first()
                distance = 0
                return(parcels_join_filt.ed_lat.iloc[0], parcels_join_filt.ed_lon.iloc[0], distance)
            else:
                return(find_nearest(lat,lon,parcels_join_filt))

    print('Run distance function and get updated information')
    locations['updated_geo'] = locations.apply(lambda x: get_updated_geo(x.parcel_id, x.f_lat, x.f_lon), axis=1)

    print("Save updated location file to data")
    locations.to_csv('./data/outputs/locations_updated.csv')

def get_metrics():

    print('Load Data')
    locations_updated = pd.read_csv('./data/outputs/locations_updated.csv')

    print('Clean updated locations')
    # Original method did not work, forced to move to get_metrics stage
    split_df = locations_updated['updated_geo'].str.split(',', expand = True)
    locations_updated['updated_lat'] = split_df[[0]]
    locations_updated['updated_lon'] = split_df[[1]]
    locations_updated['distance_km'] = split_df[[2]]
    # locations['updated_lat'] = locations['updated_geo'].str.split(',', expand = True)[0]
    # locations['updated_lon'] = locations['updated_geo'].str.split(',', expand = True)[1]
    # locations['distance_km'] = locations['updated_geo'].str.split(',', expand = True)[2]

    locations_updated['updated_lat'] = locations_updated['updated_lat'].str.replace("(","")
    locations_updated['distance_km'] = locations_updated['distance_km'].str.replace(")","")

    locations_updated['updated_lat'] = pd.to_numeric(locations_updated['updated_lat'])
    locations_updated['updated_lon'] = pd.to_numeric(locations_updated['updated_lon'])
    locations_updated['distance_km'] = pd.to_numeric(locations_updated['distance_km'])

    print("Save updated location file to data")
    locations_updated.to_csv('./data/outputs/locations_updated.csv')

    print('Calculate Metrics')
    # 1. On average, how far are original geolocation moved

    #locations_updated['distance_km'] = pd.to_numeric(locations_updated['distance_km'])
    avg_distance_moved = locations_updated['distance_km'].mean()
    min_distance_moved = locations_updated['distance_km'].min()
    max_distance_moved = locations_updated['distance_km'].max()

    # 2. How many points with too little information to move anywhere?
    # Only points without an original lat/long and a parcel without a matching building could not be moved
    total_locations = locations_updated.shape[0]
    change_locations = locations_updated.dropna(subset=['updated_geo']).shape[0]
    no_change_locations = total_locations - change_locations

    print('Create dataframe for metrics and save')
    data = [[avg_distance_moved, min_distance_moved, max_distance_moved, no_change_locations]]
    metrics = pd.DataFrame(data, columns=['avg_distance_moved', 'min_distance_moved', 'max_distance_moved', 'no_change_locations'])
    metrics.to_csv('./data/outputs/metrics.csv')

def get_mapping():

    print('Load Data')
    locations = pd.read_csv('./data/outputs/locations_updated.csv')
    buildings = gpd.read_file('./data/ms_hinds_buildings.json')

    print('Load mapping config')
    with open('./data/keplergl_config.pkl', 'rb') as f:
        new_config = pickle.load(f)

    map_2 = KeplerGl(data={"buildings": buildings[['geometry']], 
                       "orig_locations": locations[['f_lat', 'f_lon', 'f_addr1']], 
                      "new_locations": locations[['updated_lat', 'updated_lon', 'f_addr1']], 
                      "line_points": locations[['f_lat', 'f_lon', 'updated_lat', 'updated_lon', 'f_addr1']]}, config=new_config)
    
    map_2.save_to_html(data={"buildings": buildings[['geometry']], 
                       "orig_locations": locations[['f_lat', 'f_lon', 'f_addr1']], 
                      "new_locations": locations[['updated_lat', 'updated_lon', 'f_addr1']], 
                      "line_points": locations[['f_lat', 'f_lon', 'updated_lat', 'updated_lon', 'f_addr1']]}, config=new_config, file_name='./data/outputs/newgeo_map.html')


with DAG(
    dag_id='autoetl-coordinates',
    start_date=datetime(2022, 5, 28),
    schedule_interval=None
) as dag:

    get_input_data = PythonOperator(
        task_id='get_input_data',
        python_callable=get_input_data,
        provide_context=True,
        dag=dag
    )

    get_new_geo = PythonOperator(
        task_id='get_new_geo',
        python_callable=get_new_geo,
        provide_context=True,
        dag=dag
    )

    get_metrics = PythonOperator(
        task_id='get_metrics',
        python_callable=get_metrics,
        provide_context=True,
        dag=dag
    )

    get_mapping = PythonOperator(
        task_id='get_mapping',
        python_callable=get_mapping,
        provide_context=True,
        dag=dag
    )

get_input_data >> get_new_geo >> get_metrics >> get_mapping