from data_preperation.data_preprocessing import DataPreprocessing
# import pandas as pd
# import numpy as np


class GeneticAlgorithm:
    def __init__(self, data: DataPreprocessing) -> None:
        pass

    def test(self):
        pass

training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"
processed_data = DataPreprocessing(training_data)
print(processed_data.processed_dataframe.head())
print('twoja stara')