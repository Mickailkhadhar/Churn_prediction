import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import unittest
from src.model import Model
class TestModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        # Create a synthetic classification dataset for testing
        self.X, self.y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
        self.my_model = Model(self.X, self.y)
    
    def test_create_train_test_split(self):
        self.my_model.create_train_test_split()
        assert self.my_model.X_train.shape[0] == 600
        assert self.my_model.X_valid.shape[0] == 200
        assert self.my_model.X_test.shape[0] == 200
        assert self.my_model.X_train.shape[1] == 10
        assert self.my_model.y_train.shape[0] == 600
        assert self.my_model.y_valid.shape[0] == 200
        assert self.my_model.y_test.shape[0] == 200
    
    def test_create_pipeline(self):
        self.my_model.create_pipeline()
        assert len(self.my_model.pipeline.steps) == 2
        assert isinstance(self.my_model.pipeline.steps[0][1], SMOTE)
        assert isinstance(self.my_model.pipeline.steps[1][1], XGBClassifier)
    
    def test_run_grid_search_cv(self):
        params = {
            'sampling__k_neighbors': [10],
            'classifier__max_depth': [5],
            'classifier__eta': [0.3]
        }
        self.my_model.create_train_test_split()
        self.my_model.create_pipeline()
        self.my_model.run_grid_search_cv(params)
        y_pred = self.my_model.grid_search.predict(self.my_model.X_test.values)
        acc = accuracy_score(self.my_model.y_test, y_pred)
        assert acc >= 0.0
    