import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import plotly.graph_objects as go
import seaborn as sns
from src.analyze import Analyzer()
import unittest
class TestAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Create a synthetic dataset for testing
        data = {'customer_id': ['10', '11', '12', '13', '14'],
                'tenure': [6, 25, 45, 27, 40],
                'avg_days_btn_order': [40, 25, 6, 350 , 173],
                'churn': [0, 1, 0, 0, 1]}
        self.df = pd.DataFrame(data)
        self.my_analyzer = Analyzer()
    
    def test_get_pie_chart(self):
        self.my_analyzer.get_pie(self.df, 'churn', 'Churn  Distribution')
        # Ensure that the figure is of the expected type
        assert isinstance(self.my_analyzer.fig, go.Figure)
    
    def test_get_boxplot(self):
        self.my_analyzer.get_boxplot(self.df, 'tenure', 'churn', 'Churn x Tenure distrib')
        # Ensure that the figure is of the expected type
        assert isinstance(self.my_analyzer.fig, go.Figure)
    
    def test_get_barplot(self):
        # Test data preparation
        data = {'customer_id': ['10', '11', '12', '13', '14'],
                'tenure': [6, 17, 28, 35, 65],
                'avg_days_btn_order': [40, 25, 6, 350 , 173],
                'churn': [0, 1, 0, 0, 1]}
        df = pd.DataFrame(data)
        df['yearly_tenure'] = pd.cut(df['tenure'], bins=[0,12,24,36,48,60,72], 
                                    labels=['1y', '2y', '3y', '4y', '5y', '6y'])
                                    
        self.my_analyzer.get_barplot(df, 'yearly_tenure', 100*df.churn, 'Does tenure determine churn ?+')
        # Ensure that the figure is of the expected type
        assert isinstance(self.my_analyzer.fig, go.Figure)
