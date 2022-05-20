import pandas as pd
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

def day_time_cls(x):
    dawn = datetime.time(6,0,0)
    noon = datetime.time(12,0,0)
    dusk = datetime.time(18,0,0)
    midnight = datetime.time(0,0,0)
    if noon >=x.time() >= dawn:
        daytime = 'morning'
    elif dusk >= x.time() >= noon:
        daytime = 'afternoon'
    elif dawn >= x.time() >= midnight:
        daytime = 'night'
    else:
        daytime = 'evening'
    return daytime

def preprocess_train(file):
    """
    Special module in charge of importing and preprocessing the dataset
    """
    data = pd.read_csv(file)
    data = data.drop(columns=['trip_id','bike_id'])
    # Data cleaning
    data.dropna(subset=['passholder_type'],inplace=True)
    data.dropna(subset=['start_lat','start_lon','end_lat','end_lon','plan_duration'],inplace=True)

    # Data transformation
    data_t = data.copy()
    data_t['start_time'] = pd.to_datetime(data_t['start_time'])
    data_t['end_time'] = pd.to_datetime(data_t['end_time'])
    data_t['day_time'] = data_t['start_time'].apply(lambda x: day_time_cls(x))
    data_t.drop(columns='plan_duration',inplace=True)

    # Data preprocess
    X = data_t.drop(columns='passholder_type')
    y = data_t.passholder_type
    X['end_day_time'] = X.end_time.apply(lambda x: day_time_cls(x))    # Classify the end time
    # Extract seasonality of the trip
    X['year'] = X.start_time.apply(lambda x: x.year)
    X['month'] = X.start_time.apply(lambda x: x.month)
    X['weekday'] = X.start_time.apply(lambda x: x.weekday())
    X.drop(columns=['start_time','end_time'],inplace=True)
    X['trip_route_category'] = X.trip_route_category.astype('category')    # Set the dtype to category 
    X['trip_route_category'] = X.trip_route_category.cat.codes
    daytime_enc_dic = {'morning':0, 'afternoon':1, 'evening': 2, 'night':3}
    X['day_time_ord'] = X.day_time.map(daytime_enc_dic)
    X['end_day_time_ord'] = X.end_day_time.map(daytime_enc_dic)
    X['year'] = X.year.apply(lambda x: x-2017)
    X.drop(columns=['day_time','end_day_time'],inplace=True)
    X['start_station'] = X.start_station.astype('category')
    X['start_station'] = X.start_station.cat.codes
    X['end_station'] = X.end_station.astype('category')
    X['end_station'] = X.end_station.cat.codes
    # Target encoding
    LE = LabelEncoder().fit(y)
    y = LE.transform(y)
    y_df = pd.Series(y,name='target')
    # PCA transformation
    pca = PCA()
    pca.fit(X)
    components = pca.transform(X)
    components_df = pd.DataFrame(components, columns=['V'+str(i+1) for i in range(13) ])
    input_val = components_df.copy()
    output_val = y_df.copy()
    return input_val, output_val

def preprocess_test(file):
    """
    Special module in charge of importing and preprocessing the dataset
    """
    data = pd.read_csv(file)
    trip_id = data.trip_id
    data = data.drop(columns=['trip_id','bike_id'])
    # Data cleaning
    data.start_lat.fillna(value=data.start_lat.mean(),inplace=True)
    data.start_lon.fillna(value=data.start_lon.mean(),inplace=True)
    data.end_lon.fillna(value=data.end_lat.mean(),inplace=True)
    data.end_lat.fillna(value=data.end_lat.mean(),inplace=True)

    # Data transformation
    data_t = data.copy()
    data_t['start_time'] = pd.to_datetime(data_t['start_time'])
    data_t['end_time'] = pd.to_datetime(data_t['end_time'])
    data_t['day_time'] = data_t['start_time'].apply(lambda x: day_time_cls(x))

    # Data preprocess
    X = data_t.copy()
    X['end_day_time'] = X.end_time.apply(lambda x: day_time_cls(x))    # Classify the end time
    # Extract seasonality of the trip
    X['year'] = X.start_time.apply(lambda x: x.year)
    X['month'] = X.start_time.apply(lambda x: x.month)
    X['weekday'] = X.start_time.apply(lambda x: x.weekday())
    X.drop(columns=['start_time','end_time'],inplace=True)
    X['trip_route_category'] = X.trip_route_category.astype('category')    # Set the dtype to category 
    X['trip_route_category'] = X.trip_route_category.cat.codes
    daytime_enc_dic = {'morning':0, 'afternoon':1, 'evening': 2, 'night':3}
    X['day_time_ord'] = X.day_time.map(daytime_enc_dic)
    X['end_day_time_ord'] = X.end_day_time.map(daytime_enc_dic)
    X['year'] = X.year.apply(lambda x: x-2017)
    X.drop(columns=['day_time','end_day_time'],inplace=True)
    X['start_station'] = X.start_station.astype('category')
    X['start_station'] = X.start_station.cat.codes
    X['end_station'] = X.end_station.astype('category')
    X['end_station'] = X.end_station.cat.codes
    # PCA transformation
    pca = PCA()
    pca.fit(X)
    components = pca.transform(X)
    components_df = pd.DataFrame(components, columns=['V'+str(i+1) for i in range(13) ])
    input_val = components_df.copy()
    return input_val,trip_id
