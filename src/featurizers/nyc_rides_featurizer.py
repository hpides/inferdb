import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from copy import deepcopy
from numpy import percentile
from sklearn.linear_model import LinearRegression
from geographiclib.geodesic import Geodesic
from geographiclib.constants import Constants
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

class NYC_Featurizer(BaseEstimator, TransformerMixin):

    def __init__(self, depth) -> None:
        super().__init__()

        self.depth = depth

        if self.depth == 'deep':
            self.cat_features = [
                                    
                                 ]
            
            self.num_features = [
                                'pickup_weekday', 'is_weekend', 'pickup_weekofyear', 'pickup_hour', 'pickup_minute', 
                                # 'pickup_dt', 'pickup_week_hour',
                                'srccounty', 'dstcounty', 'vendor_id',
                                'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                                
                                
                                'distance', 'duration', 'motorway', 'trunk', 'primarie', 'secondary', 'tertiary', 'unclassified', 'residential', 
                                'ntrafficsignals', 'ncrossing', 'nstop', 'nintersection',
                    
                                # 'pickup_cluster', 'dropoff_cluster',
                                'arc', 'pickup_bearing', 'dropoff_bearing',
                                'pickup_pca0', 'pickup_pca1', 'dropoff_pca0', 'dropoff_pca1', 'pca_manhattan'

                                

                                # , 'avg_distance'
                                # , 'avg_travel_time'
                                # , 'avg_cnt_of_steps'
                                # , 'cnt'
                                # , 'avg_trip_duration'
                                # , 'avg_speed'
                                , 'geo_distance'
                                # , 'freq_dist'
                                ]
        else:
            self.cat_features = [
                                    
                                ]
            self.num_features = [
                                   'srcCounty', 'dstCounty','pickup_weekday', 'is_weekend', 'vendor_id', 'passenger_count', 
                                   'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                                    'pickup_weekofyear', 'pickup_hour', 'pickup_minute', 'pickup_dt', 'pickup_week_hour',
                                    'distance'

                                ]
        
        self.infer_db_subset = [
                                'vendor_id',
                                'passenger_count', 
                                'pickup_longitude', 
                                'pickup_latitude', 
                                'dropoff_longitude', 
                                'dropoff_latitude',
                                'pickup_weekday', 
                                'pickup_hour', 
                                'is_weekend',
                                'distance'
                                ]
    
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

        geo = Geodesic(Constants.WGS84_a, Constants.WGS84_f)
        
        dict = geo.Inverse(pickup_lat, pickup_long, dropoff_lat, dropoff_long)

        return dict['s12'] / 1000
    
    def cal_arc(self, pickup_lat, pickup_long, dropoff_lat, dropoff_long):

        geo = Geodesic(Constants.WGS84_a, Constants.WGS84_f)

        dict = geo.Inverse(pickup_lat, pickup_long, dropoff_lat, dropoff_long)

        return dict['a12']
    
    def cal_pickup_bearing(self, pickup_lat, pickup_long, dropoff_lat, dropoff_long):

        geo = Geodesic(Constants.WGS84_a, Constants.WGS84_f)

        dict = geo.Inverse(pickup_lat, pickup_long, dropoff_lat, dropoff_long)

        if dict['azi1'] > 0 and dict['azi1'] <= 90:
            pickup_bearing_direction = 'NE' 
        elif dict['azi1'] > 90 and dict['azi1'] <= 180:
            pickup_bearing_direction = 'NW'
        elif dict['azi1'] > 180 and dict['azi1'] <= 270:
            pickup_bearing_direction = 'SW'
        else:
            pickup_bearing_direction = 'SE'

        return pickup_bearing_direction
    
    def cal_pickup_bearing_degrees(self, pickup_lat, pickup_long, dropoff_lat, dropoff_long):

        geo = Geodesic(Constants.WGS84_a, Constants.WGS84_f)

        dict = geo.Inverse(pickup_lat, pickup_long, dropoff_lat, dropoff_long)

        return dict['azi1']
    
    def cal_dropoff_bearing(self, pickup_lat, pickup_long, dropoff_lat, dropoff_long):

        geo = Geodesic(Constants.WGS84_a, Constants.WGS84_f)

        dict = geo.Inverse(pickup_lat, pickup_long, dropoff_lat, dropoff_long)
        
        if dict['azi2'] > 0 and dict['azi2'] <= 90:
            dropoff_bearing_direction = 'NE' 
        elif dict['azi2'] > 90 and dict['azi2'] <= 180:
            dropoff_bearing_direction = 'NW'
        elif dict['azi2'] > 180 and dict['azi2'] <= 270:
            dropoff_bearing_direction = 'SW'
        else:
            dropoff_bearing_direction = 'SE'

        return dropoff_bearing_direction
    
    def cal_dropoff_bearing_degrees(self, pickup_lat, pickup_long, dropoff_lat, dropoff_long):

        geo = Geodesic(Constants.WGS84_a, Constants.WGS84_f)

        dict = geo.Inverse(pickup_lat, pickup_long, dropoff_lat, dropoff_long)

        return dict['azi2']
    
    def learn_iqr(self, feature):

        q25, q75 = percentile(feature, 25), percentile(feature, 75)
        iqr = q75 - q25
        cut_off = iqr * 3
        lower, upper = q25 - cut_off, q75 + cut_off

        return lower, upper
    
    def learn_pca(self, X):
    
        coords = np.vstack((X[['pickup_latitude', 'pickup_longitude']].values,
                        X[['dropoff_latitude', 'dropoff_longitude']].values))
        
        pca = PCA(n_components=2).fit(coords)

        return pca
    
    def learn_clustering(self, X):

        coords = np.vstack((X[['pickup_latitude', 'pickup_longitude']].values,
                        X[['dropoff_latitude', 'dropoff_longitude']].values))

        # sample_ind = np.random.permutation(len(coords))[:500000]
        # kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
        kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords)

        return kmeans

    def fit(self, X, y):

        X_ = deepcopy(X)
        X_.reset_index(drop=True, inplace=True)

        self.imputers = {}
        self.outlier_imputers = {}
        self.outlier_training_features = {}
        self.outlier_boundaries = {}
        self.columns = X_.columns
        for feature in self.columns:
            #### Impute
            if feature in ('pickup_datetime', 'dropoff_datetime'):
                imputer = self.impute(X_[feature], 'datetime')
                self.imputers[feature] = imputer
                X_[feature] = X_[feature].fillna(imputer)
            elif is_string_dtype(X_[feature]):
                imputer = self.impute(X_[feature], 'string')
                self.imputers[feature] = imputer
                X_[feature] = X_[feature].fillna(imputer)
            else:
                imputer = self.impute(X_[feature], 'numeric')
                self.imputers[feature] = imputer
                X_[feature] = X_[feature].fillna(imputer)
            #### Encode

        ##### Freq and speed Mappers
        X_['pickup_datetime'] = pd.to_datetime(X_['pickup_datetime'])

        ######## DISTANCE
        X_['geo_distance'] = X_.apply(lambda x: self.cal_distance(x['pickup_latitude'],x['pickup_longitude'],x['dropoff_latitude'],x['dropoff_longitude'] ), axis=1)
        ##################
        
        if self.depth == 'deep':

            for feature in [i for i in self.num_features if i in ('passenger_count', 'distance', 'geo_distance')]:

                lower, upper = self.learn_iqr(X_[feature].to_numpy())
                self.outlier_boundaries[feature] = (lower, upper)
                outlier_indices = np.nonzero((X_[feature].to_numpy() < lower) | (X_[feature].to_numpy() > upper))

                X_train = X_.iloc[outlier_indices[0], [idx for idx, i in enumerate(list(X_)) if i != feature and (i in self.num_features or i in self.cat_features)]]
                y_train = X_.iloc[outlier_indices[0], list(X_).index(feature)].to_numpy()

                
                training_features = [i for i in list(X_) if i != feature and i in self.num_features]
                self.outlier_training_features[feature] = training_features
                reg = LinearRegression(n_jobs=-1)
                
                reg.fit(X_train[training_features], y_train)

                self.outlier_imputers[feature] = (reg.intercept_, reg.coef_)

            ########## DATETIME FEATURES:

            X_['pickup_weekday'] = X_['pickup_datetime'].dt.weekday
            X_['pickup_hour'] = X_['pickup_datetime'].dt.hour

            ################################
            ######### CLUSTERING
            ###############################

            self.kmeans = self.learn_clustering(X_)

            X_['pickup_cluster'] = self.kmeans.predict(X_[['pickup_latitude', 'pickup_longitude']].values)
            X_['dropoff_cluster'] = self.kmeans.predict(X_[['dropoff_latitude', 'dropoff_longitude']].values)

            ################################
            ######### GEO SPATIAL AGGREGATION
            ###############################


            X_['speed'] = X_['geo_distance'] / y

            speed_df = X_.groupby(['pickup_cluster', 'dropoff_cluster'])['speed'].agg(['mean'])
            self.route_avg_speed = speed_df.to_dict('index')

            clusters_avg_trip_duration_df = deepcopy(X_)
            clusters_avg_trip_duration_df['trip_duration'] = y
            agg_clusters_avg_trip_duration_df = clusters_avg_trip_duration_df.groupby(['pickup_cluster', 'dropoff_cluster'])['trip_duration'].agg(['mean'])
            self.route_avg_trip_duration = agg_clusters_avg_trip_duration_df.to_dict('index')

            traffic_df = X_.groupby(['pickup_cluster', 'dropoff_cluster'])['id'].agg(['count'])
            self.route_freq_mappers = traffic_df.to_dict('index')

            self.pca = self.learn_pca(X_)

            ################################
            ######### FASTEST ROUTES AGG
            ###############################

            steps_df = X_.groupby(['pickup_cluster', 'dropoff_cluster'])['number_of_steps'].agg(['mean'])
            self.route_avg_steps = steps_df.to_dict('index')

            travel_time_df = X_.groupby(['pickup_cluster', 'dropoff_cluster'])['total_travel_time'].agg(['mean'])
            self.route_avg_travel_time = travel_time_df.to_dict('index')

            distance_df = X_.groupby(['pickup_cluster', 'dropoff_cluster'])['total_distance'].agg(['mean'])
            self.route_avg_distance = distance_df.to_dict('index')

            mappers_df = X_.groupby(['pickup_cluster', 'dropoff_cluster'])[['total_distance', 'total_travel_time', 'number_of_steps', 'id', 'duration', 'speed']].agg({
                                                                                                                                                                                    'total_distance':['mean'],
                                                                                                                                                                                    'total_travel_time':['mean'],
                                                                                                                                                                                    'number_of_steps':['mean'],
                                                                                                                                                                                    'id':['count'],
                                                                                                                                                                                    'duration':['mean'],
                                                                                                                                                                                    'speed':['mean']
                                                                                                                                                                                })
            
            
            self.cluster_mappers = mappers_df

        return self
    
    def get_feature_names(self):

        return self.feature_names
        
    def transform(self, X):

        X_ = deepcopy(X)
        if isinstance(X_, pd.Series):
            X_ = X_.to_frame().transpose()
         
        # for feature in self.columns:
        #     X_[feature] = X_[feature].fillna(self.imputers[feature])

        X_['pickup_datetime'] = pd.to_datetime(X_['pickup_datetime'])
        X_['pickup_weekday'] = X_['pickup_datetime'].dt.weekday
        X_['pickup_weekofyear'] = X_['pickup_datetime'].dt.isocalendar().week
        X_['pickup_hour'] = X_['pickup_datetime'].dt.hour
        X_['pickup_minute'] = X_['pickup_datetime'].dt.minute
        X_['pickup_dt'] = (X_['pickup_datetime'] - X_['pickup_datetime'].min()).dt.total_seconds()
        X_['pickup_week_hour'] = X_['pickup_weekday'] * 24 + X_['pickup_hour']
        X_['geo_distance'] = X_.apply(lambda x: self.cal_distance(x['pickup_latitude'],x['pickup_longitude'],x['dropoff_latitude'],x['dropoff_longitude'] ), axis=1)
        X_['is_weekend'] = X_.apply(lambda x: 1 if x['pickup_weekday'] in (5, 6) else 0, axis=1)
        

        if self.depth == 'deep':
            X_['arc'] = X_.apply(lambda x: self.cal_arc(x['pickup_latitude'],x['pickup_longitude'],x['dropoff_latitude'],x['dropoff_longitude'] ), axis=1)
            X_['pickup_bearing'] = X_.apply(lambda x: self.cal_pickup_bearing_degrees(x['pickup_latitude'],x['pickup_longitude'],x['dropoff_latitude'],x['dropoff_longitude'] ), axis=1)
            X_['dropoff_bearing'] = X_.apply(lambda x: self.cal_dropoff_bearing_degrees(x['pickup_latitude'],x['pickup_longitude'],x['dropoff_latitude'],x['dropoff_longitude'] ), axis=1)

            def get_route_freq(X_):

                try:
                    value = self.route_freq_mappers[(X_['pickup_cluster'], X_['dropoff_cluster'])]['count']
                    if np.isfinite(value): 
                        return value
                    else:
                        return 0
                except KeyError:
                    return 0
            
            def get_route_avg_speed(X_):

                try:
                    value = self.route_avg_speed[(X_['pickup_cluster'], X_['dropoff_cluster'])]['mean']
                    if np.isfinite(value): 
                        return value
                    else:
                        return 0
                except KeyError:
                    return 0
            
            def get_route_avg_duration(X_):
                try:
                    value = self.route_avg_trip_duration[(X_['pickup_cluster'], X_['dropoff_cluster'])]['mean']
                    if np.isfinite(value): 
                        return value
                    else:
                        return 0
                except KeyError:
                    return 0
            
            def get_route_avg_steps(X_):
                try:
                    value = self.route_avg_steps[(X_['pickup_cluster'], X_['dropoff_cluster'])]['mean']
                    if np.isfinite(value): 
                        return value
                    else:
                        return 0
                except KeyError:
                    return 0 
            
            def get_route_avg_travel_time(X_):
                try:
                    value = self.route_avg_travel_time[(X_['pickup_cluster'], X_['dropoff_cluster'])]['mean']
                    if np.isfinite(value): 
                        return value
                    else:
                        return 0
                except KeyError:
                    return 0 
            
            def get_route_avg_distance(X_):
                try:
                    value = self.route_avg_distance[(X_['pickup_cluster'], X_['dropoff_cluster'])]['mean']
                    if np.isfinite(value): 
                        return value
                    else:
                        return 0
                except KeyError:
                    return 0 
            
            def predict_outlier_passenger(X_):
                
                instance = X_.to_frame().transpose()
                X_test = instance[self.outlier_training_features['passenger_count']].to_numpy()[0]
                pred = self.outlier_imputers['passenger_count'][0] + np.dot(self.outlier_imputers['passenger_count'][1], X_test)

                return pred
            
            def predict_outlier_distance(X_):
                
                instance = X_.to_frame().transpose()
                X_test = instance[self.outlier_training_features['distance']].to_numpy()[0]
                pred = self.outlier_imputers['distance'][0] + np.dot(self.outlier_imputers['distance'][1], X_test)

                return pred
            
            def predict_outlier_geo_distance(X_):

                instance = X_.to_frame().transpose()
                X_test = instance[self.outlier_training_features['geo_distance']].to_numpy()[0]
                pred = self.outlier_imputers['geo_distance'][0] + np.dot(self.outlier_imputers['geo_distance'][1], X_test)

                return pred
            
            ########### Outlier Imputation

            X_['passenger_count'] = X_.apply(lambda x: predict_outlier_passenger(x) if x['passenger_count'] > self.outlier_boundaries['passenger_count'][1] or x['passenger_count'] < self.outlier_boundaries['passenger_count'][0] else x['passenger_count'], axis=1)
            X_['distance'] = X_.apply(lambda x: predict_outlier_distance(x) if x['distance'] > self.outlier_boundaries['distance'][1] or x['distance'] < self.outlier_boundaries['distance'][0] else x['distance'], axis=1)
            X_['geo_distance'] = X_.apply(lambda x: predict_outlier_geo_distance(x) if x['geo_distance'] > self.outlier_boundaries['geo_distance'][1] or x['geo_distance'] < self.outlier_boundaries['geo_distance'][0] else x['geo_distance'], axis=1)

            ###########

            X_['pickup_cluster'] = self.kmeans.predict(X_[['pickup_latitude', 'pickup_longitude']].values)
            X_['dropoff_cluster'] = self.kmeans.predict(X_[['dropoff_latitude', 'dropoff_longitude']].values)

            X_['cnt'] = X_.apply(lambda x: get_route_freq(x), axis=1)
            X_['avg_speed'] = X_.apply(lambda x: get_route_avg_speed(x), axis=1)
            X_['avg_trip_duration'] = X_.apply(lambda x: get_route_avg_duration(x), axis=1)
            X_['freq_dist'] = X_.apply(lambda x: x['cnt'] * x['geo_distance'], axis=1)
            X_['avg_cnt_of_steps'] = X_.apply(lambda x: get_route_avg_steps(x), axis=1)
            X_['avg_travel_time'] = X_.apply(lambda x: get_route_avg_travel_time(x), axis=1)
            X_['avg_distance'] = X_.apply(lambda x: get_route_avg_distance(x), axis=1)

            X_['pickup_pca0'] = self.pca.transform(X_[['pickup_latitude', 'pickup_longitude']].values)[:, 0]
            X_['pickup_pca1'] = self.pca.transform(X_[['pickup_latitude', 'pickup_longitude']].values)[:, 1]
            X_['dropoff_pca0'] = self.pca.transform(X_[['dropoff_latitude', 'dropoff_longitude']].values)[:, 0]
            X_['dropoff_pca1'] = self.pca.transform(X_[['dropoff_latitude', 'dropoff_longitude']].values)[:, 1]
            X_['pca_manhattan'] = np.abs(X_['dropoff_pca1'] - X_['pickup_pca1']) + np.abs(X_['dropoff_pca0'] - X_['pickup_pca0'])

        X_.rename(columns={'primary':'primarie', 'nTrafficSignals': 'ntrafficsignals', 'nCrossing':'ncrossing', 'nStop':'nstop', 'nIntersection':'nintersection','srcCounty':'srccounty', 'dstCounty':'dstcounty'}, inplace=True)

        self.feature_names = list(X_)
        self.features_to_keep = [i for i in self.feature_names if i in self.cat_features or i in self.num_features]

        return X_.loc[:, self.num_features]

    def transform_for_inferdb(self, X):

        X_ = deepcopy(X)
        if isinstance(X_, pd.Series):
            X_ = X_.to_frame().transpose()
         
        for feature in self.columns:
            X_[feature] = X_[feature].fillna(self.imputers[feature])

        X_['pickup_datetime'] = pd.to_datetime(X_['pickup_datetime'])
        X_['pickup_weekday'] = X_['pickup_datetime'].dt.weekday
        X_['pickup_hour'] = X_['pickup_datetime'].dt.hour
        X_['is_weekend'] = X_.apply(lambda x: 1 if x['pickup_weekday'] in (5, 6) else 0, axis=1)

        return X_.loc[:, self.infer_db_subset]


