import numpy as np

import data_preperation.data_preprocessing as data_prep
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler


class GATesting:
    def __init__(self, test_df_path: str, ga_results_path: str):
        self.test_data_processed = data_prep.DataPreprocessing(test_df_path)
        self.test_df = self.test_data_processed.processed_dataframe.drop(labels=['id'], axis=1)
        self.ga_results = pd.read_csv(ga_results_path)
        self.positive = 0
        self.false_attack = 0
        self.false_normal = 0
        self.negative = 0

        self.test_df_unlabeled = self.test_df.drop(labels=['label'], axis=1)

    def checkIfAnomaly(self, idx):
        # TODO: how to do that
        for index, row in self.ga_results.iterrows():
            pass

    def testGA(self):
        for i in range(70000, 80800):
            self.checkIfAnomaly(i)
        print('positive', self.positive)
        print('false positive', self.false_attack)
        print('negative', self.negative)
        print('false negative', self.false_normal)
        print('total = ', self.positive + self.false_attack + self.negative + self.false_normal)
        print('accuracy = ', (self.positive + self.negative) / (self.positive + self.false_attack +
                                                                self.negative + self.false_normal) * 100, '%')

