# imports
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import *
from TaxiFareModel.utils import *
from TaxiFareModel.data import *
import pandas as pd
import numpy as np


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        # create distance pipeline
        dist_pipeline = Pipeline([('dist_trans', DistanceTransformer()),
                                ('robust_scaler', StandardScaler())
                                ])

        # create time pipeline
        time_pipeline = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                                ('ohe_enc', OneHotEncoder(sparse=False, handle_unknown='ignore'))
                                ])

        # column transformer
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        # create preprocessing pipeline
        preproc_pipeline = ColumnTransformer([('time', time_pipeline, time_cols),
                                          ('distance', dist_pipeline, dist_cols)
                                        ])

        # display the pipeline with model
        final_pipeline = Pipeline(steps=[('preproc_pipeline', preproc_pipeline),
                                    ('model_ridge', GradientBoostingRegressor())
                                    ])

        self.pipeline = final_pipeline

        return self


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

        return self


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""

        y_pred = self.pipeline.predict(X_test)

        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    #get data
    data = get_data(nrows=10000)
    # clean data
    data = clean_data(data)
    # set X and y
    X = data.drop(columns=['fare_amount'])
    y = data['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    score = trainer.evaluate(X_test, y_test)
    print(score)
