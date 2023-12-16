import numpy as np

import data_preperation.data_preprocessing as data_prep
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler


class GATesting:
    def __init__(self, test_df_path: str, ga_results_path: str):
        self.test_data_processed = data_prep.DataPreprocessing(test_df_path, alg_type='ga')
        self.test_df = self.test_data_processed.processed_dataframe.drop(labels=['id'], axis=1)
        # self.test_df = self.test_df.head(50000)
        self.ga_results = pd.read_csv(ga_results_path)
        self.positive = 0
        self.false_attack = 0
        self.false_normal = 0
        self.negative = 0

        self.test_df_unlabeled = self.test_df.drop(labels=['label'], axis=1)

    def checkIfAnomaly(self, d):
        # TODO: how to do that
        for index_test, row_test in self.test_df.iterrows():
            print(index_test)
            flag = 0
            for index_ga, row_ga in self.ga_results.iterrows():
                common_columns = set(row_test.index) & set(row_ga.index)
                common_elements = sum(row_test[col] == row_ga[col] for col in common_columns)

                if common_elements >= d and row_test['label'] == 1:
                    self.positive += 1
                    flag = 1
                    break
                elif common_elements >= d and row_test['label'] == 0:
                    self.false_attack += 1
                    flag = 1
                    break
            if flag == 0 and row_test['label'] == 1:
                self.false_normal += 1
            elif flag == 0 and row_test['label'] == 0:
                self.negative += 1

    def testGA(self):

        self.checkIfAnomaly(8)
        print('positive', self.positive)
        print('false positive', self.false_attack)
        print('negative', self.negative)
        print('false negative', self.false_normal)
        print('total = ', self.positive + self.false_attack + self.negative + self.false_normal)
        print('accuracy = ', (self.positive + self.negative) / (self.positive + self.false_attack +
                                                                self.negative + self.false_normal) * 100, '%')

