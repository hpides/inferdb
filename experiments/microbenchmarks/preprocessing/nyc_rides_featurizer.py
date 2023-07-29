import pandas as pd
from pandas.api.types import is_string_dtype
from geopy.distance import great_circle
from optbinning import ContinuousOptimalBinning2D
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.impute import SimpleImputer


class NYC_Featurizer(BaseEstimator, TransformerMixin):

    def __init__(self, depth) -> None:
        super().__init__()

        self.depth = depth

        if self.depth == 'deep':
            self.cat_features = ['day', 'is_weekend', 'clusters_cluster', 'month', 'pickup_hour_of_day']
            self.num_features = ['passenger_count', 'distance', 'hour', 'route_freq', 'freq_dist']
        else:
            self.cat_features = ['day', 'is_weekend', 'month']
            self.num_features = ['passenger_count', 'hour', 'distance']
    
    def impute(self, feature, dtype):

        if dtype == 'datetime':
            feature_array = np.asarray(feature).reshape(-1, 1)
            mean = (np.array(feature_array, dtype='datetime64[s]')
                    .view('i8')
                    .mean()
                    .astype('datetime64[s]'))
            return mean
        elif dtype == 'string':
            return feature.mode()[0]
        elif dtype == 'numeric':
            return feature.mean()

    def get_pickup_cluster(self, pickup_lat,pickup_long, target):
    
        encoder = ContinuousOptimalBinning2D('pickup_cluster')
        encoder.fit(pickup_lat, pickup_long, target)
    
        return encoder

    def get_dropoff_cluster(self, dropoff_lat, dropoff_long, target):
    
        encoder = ContinuousOptimalBinning2D('dropoff_cluster')
        encoder.fit(dropoff_lat, dropoff_long, target)
    
        return encoder
    
    def get_clusters_cluster(self, pickup_cluster, dropoff_cluster, target):

        encoder = ContinuousOptimalBinning2D('clusters_cluster', dtype_x='categorical', dtype_y='categorical')
        encoder.fit(pickup_cluster, dropoff_cluster, target)

        return encoder

    def time_of_day(self, time):
        if time in range(6,12):
            return 'morning'
        elif time in range(12,16):
            return 'afternoon'
        elif time in range(16,22):
            return 'evening'
        else:
            return 'late_night'
    
    def cal_distance(self, pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    
        start_coordinates=(pickup_lat,pickup_long)
        stop_coordinates=(dropoff_lat,dropoff_long)
    
        return great_circle(start_coordinates, stop_coordinates).km
    
    def fit(self, X, y):

        self.imputers = {}
        self.columns = X.columns
        for feature in self.columns:
            #### Impute
            if feature in ('pickup_datetime', 'dropoff_datetime'):
                imputer = self.impute(X[feature], 'datetime')
                self.imputers[feature] = imputer
                X[feature] = X[feature].fillna(imputer)
            elif is_string_dtype(X[feature]):
                imputer = self.impute(X[feature], 'string')
                self.imputers[feature] = imputer
                X[feature] = X[feature].fillna(imputer)
            else:
                imputer = self.impute(X[feature], 'numeric')
                self.imputers[feature] = imputer
                X[feature] = X[feature].fillna(imputer)
            #### Encode
        
        self.pickup_encoder = self.get_pickup_cluster(X['pickup_latitude'], X['pickup_longitude'], y)
        self.dropoff_encoder = self.get_dropoff_cluster(X['dropoff_latitude'], X['dropoff_longitude'], y)

        ##### Freq Mappers
        X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])
        X['day'] = X.apply(lambda x: x['pickup_datetime'].day_name(), axis=1)
        X['hour'] = X['pickup_datetime'].dt.hour
        if self.depth == 'deep':
            X['pickup_cluster'] = self.pickup_encoder.transform(X['pickup_latitude'], X['pickup_longitude'], metric='indices')
            X['dropoff_cluster'] = self.dropoff_encoder.transform(X['dropoff_latitude'], X['dropoff_longitude'], metric='indices')
            X['cluster_combination'] = X.apply(lambda x: str(x['pickup_cluster']) + '_' + str(x['dropoff_cluster']), axis=1)
            agg_df = X.groupby(['cluster_combination', 'day', 'hour'])['id'].agg(['count'])
            self.route_freq_mappers = agg_df.to_dict('index')
            self.clusters_encoder = self.get_clusters_cluster(X['pickup_cluster'], X['dropoff_cluster'], y)

        return self
        
    def transform(self, X):

        X = X.copy()
        
        if isinstance(X, pd.DataFrame): 
            for feature in self.columns:
                X[feature] = X[feature].fillna(self.imputers[feature])

            X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])
            X['day'] = X.apply(lambda x: x['pickup_datetime'].day_name(), axis=1)
            X['is_weekend'] = X.apply(lambda x: 1 if x['day'] in ('Saturday', 'Sunday') else 0, axis=1)
            X['hour'] = X['pickup_datetime'].dt.hour
            X['month'] = X['pickup_datetime'].dt.month
            X['distance'] = X.apply(lambda x: self.cal_distance(x['pickup_latitude'],x['pickup_longitude'],x['dropoff_latitude'],x['dropoff_longitude'] ), axis=1)
            

            if self.depth == 'deep':
                X['pickup_hour_of_day'] = X['hour'].apply(self.time_of_day)
                X['pickup_cluster'] = self.pickup_encoder.transform(X['pickup_latitude'], X['pickup_longitude'], metric='indices')
                X['dropoff_cluster'] = self.dropoff_encoder.transform(X['dropoff_latitude'], X['dropoff_longitude'], metric='indices')
                X['cluster_combination'] = X.apply(lambda x: str(x['pickup_cluster']) + '_' + str(x['dropoff_cluster']), axis=1)

                def get_route_freq(x):

                    try:
                        return self.route_freq_mappers[(x['cluster_combination'], x['day'], x['hour'])]['count']
                    except KeyError:
                        return 0

                X['route_freq'] = X.apply(lambda x: get_route_freq(x), axis=1)
                X['freq_dist'] = X.apply(lambda x: x['route_freq'] * x['distance'], axis=1)
                X['clusters_cluster'] = self.clusters_encoder.transform(X['pickup_cluster'], X['dropoff_cluster'], metric='indices')


                X.drop(columns=['vendor_id', 'store_and_fwd_flag','pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_datetime', 'id', 'pickup_cluster', 'dropoff_cluster', 'dropoff_datetime', 'cluster_combination'], inplace=True)
            else:
                X.drop(columns=['vendor_id', 'store_and_fwd_flag', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_datetime', 'id', 'dropoff_datetime'], inplace=True)
        elif isinstance(X, pd.Series):
            for feature in self.columns:
                if pd.isna(X[feature]):
                    X[feature] = self.imputers[feature]
                else:
                    continue

            X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])
            X['day'] = X['pickup_datetime'].day_name()
            X['is_weekend'] = 1 if X['day'] in ('Saturday', 'Sunday') else 0
            X['hour'] = X['pickup_datetime'].hour
            X['month'] = X['pickup_datetime'].month
            X['distance'] = self.cal_distance(X['pickup_latitude'], X['pickup_longitude'], X['dropoff_latitude'], X['dropoff_longitude'])
            

            if self.depth == 'deep':
                X['pickup_hour_of_day'] = self.time_of_day(X['hour'])
                np.asarray(X['pickup_latitude']).reshape(1, -1)
                X['pickup_cluster'] = self.pickup_encoder.transform(np.asarray(X['pickup_latitude']).reshape(1, -1), np.asarray(X['pickup_longitude']).reshape(1, -1), metric='indices')[0][0]
                X['dropoff_cluster'] = self.dropoff_encoder.transform(np.asarray(X['dropoff_latitude']).reshape(1, -1), np.asarray(X['dropoff_longitude']).reshape(1, -1), metric='indices')[0][0]
                X['cluster_combination'] = str(X['pickup_cluster']) + '_' + str(X['dropoff_cluster'])

                def get_route_freq(x):

                    try:
                        return self.route_freq_mappers[(x['cluster_combination'], x['day'], x['hour'])]['count']
                    except KeyError:
                        return 0

                X['route_freq'] = get_route_freq(X)
                X['freq_dist'] = X['route_freq'] * X['distance']
                X['clusters_cluster'] = self.clusters_encoder.transform(np.asarray(X['pickup_cluster']).reshape(1, -1), np.asarray(X['dropoff_cluster']).reshape(1, -1), metric='indices')[0][0]


                X.drop(labels=['vendor_id', 'store_and_fwd_flag','pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_datetime', 'id', 'pickup_cluster', 'dropoff_cluster', 'dropoff_datetime', 'cluster_combination'], inplace=True)
            else:
                X.drop(labels=['vendor_id', 'store_and_fwd_flag','pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_datetime', 'id', 'dropoff_datetime'], inplace=True)

            X = pd.DataFrame([X.tolist()], columns=X.index)
        return X
