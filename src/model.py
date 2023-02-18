import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost
from xgboost import XGBClassifier
import imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

class Model:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def create_train_test_split(self, test_size = 0.2, random_state = 12):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = random_state)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, test_size = 0.25, random_state = random_state)

    ##getter for each set
    def get_train_set(self):
        return self.X_train, self.y_train

    def get_valid_set(self):
        return self.X_valid, self.y_valid

    def get_test_set(self):
        return self.X_test, self.y_test


    def create_pipeline(self):
        self.pipeline = Pipeline([
            ('sampling', SMOTE()),
            ('classifier', XGBClassifier())
            ])


    def run_grid_search_cv(self, params):
        self.grid_search = GridSearchCV(self.pipeline, params, cv = 5, n_jobs = -1, scoring = 'f1', refit = True)
        self.grid_search.fit(self.X_train, self.y_train)

    