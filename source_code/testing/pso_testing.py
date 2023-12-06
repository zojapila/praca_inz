import numpy as np

import data_preperation.data_preprocessing as data_prep
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class PSOTesting:
    def __init__(self, test_df_path: str, pso_results_path: str):
        self.test_data_processed = data_prep.DataPreprocessing(test_df_path)
        self.test_df = self.test_data_processed.processed_dataframe.drop(labels=['id'], axis=1)
        self.normalization()
        self.pso_results = pd.read_csv(pso_results_path).drop(labels=['label'], axis=1)
        self.test_df_unlabeled = self.test_df.drop(labels=['label'], axis=1)
        self.positive = 0
        self.false_attack = 0
        self.negative = 0
        self.false_normal = 0

    def normalization(self):
        scaler = MinMaxScaler()
        self.test_df = pd.DataFrame(scaler.fit_transform(self.test_df), columns=self.test_df.columns)

    def checkIfAnomaly(self, idx):
        flag = 0  # 1 2/ 3 4
        for index, row in self.pso_results.iterrows():
            diff = np.subtract(self.test_df_unlabeled.iloc[idx], row)
            dist = np.sum(np.power(diff, 2))
            if dist - (row['r'] ** 2) < 0:
                if self.test_df.iloc[idx]['label'] == 1:
                    # print('jest moc')
                    self.positive += 1
                    flag = 1
                    break
                else:
                    # print('no tak średnio')
                    self.false_attack += 1
                    flag = 2
                    break
            else:
                if self.test_df.iloc[idx]['label'] == 1:
                    flag = 3
                    # self.false_normal += 1
                # else:
                    # self.negative += 1
        if flag == 0:
            self.negative += 1
        elif flag == 3:
            self.false_normal += 1

    def testPSO(self):
        # print('test df\n', self.test_df.head())
        # print('pso results\n', self.pso_results.head())
        for i in range(self.test_df.shape[0]):
            self.checkIfAnomaly(i)
        print('positive', self.positive)
        print('false positive', self.false_attack)
        print('negative', self.negative)
        print('false negative', self.false_normal)
        print('total = ', self.positive + self.false_attack + self.negative + self.false_normal)
        print('accuracy = ', (self.positive + self.negative) / (self.positive + self.false_attack +
                                                                self.negative + self.false_normal) * 100, '%')




