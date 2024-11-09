import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):

        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        """
        print(type(train_data))
        for col in train_data.select_dtypes(include='object').columns:  # Selects only categorical (object) columns
            print(f"Column: {col}")
            print(train_data[col].value_counts())
            print("\n")

        for col in test_data.select_dtypes(include='object').columns:  # Selects only categorical (object) columns
            print(f"Column: {col}")
            print(test_data[col].value_counts())
            print("\n")
        """
        le=LabelEncoder()
        train_data['type'] = le.fit_transform(train_data['type'])
        test_data['type'] = le.fit_transform(test_data['type'])

        imputer = SimpleImputer(strategy='mean')
        #print(train_data.isna().sum())
        #print(test_data.isna().sum())
        train_data= pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
        test_data= pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        
        print("bbbbbbbbbbbbbb")

        """
        print(train_x)
        print("jjjjjjjjjjjjjjjjjjj")
        print(test_x)
        print("nnnnnnnnnnnnnnnnnn")
        print(train_y)
        print("nnnmmmmmmmmmmmmmmmmmmmm")
        print(test_y)
        """

        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)

        lr.fit(train_x, train_y)
        #model = HistGradientBoostingRegressor()
        #model.fit(train_x, train_y)
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))