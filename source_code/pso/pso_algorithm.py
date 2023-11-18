import random

import pandas as pd
from data_preperation.data_preprocessing import DataPreprocessing
from sklearn.preprocessing import MinMaxScaler


class ParticleSwarmOptimization:
    def __init__(self, data: DataPreprocessing, population_size: int = 100):
        self.population_size = population_size
        self.data = data.processed_dataframe
        self.data = self.data.drop(labels=["id"], axis=1)
        self.unlabeled_data = None
        self.database_size = self.data.shape[0]
        self.population = []

    def normalization(self):
        scaler = MinMaxScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        self.unlabeled_data = self.data.drop(labels=['label'], axis=1)

    def generateInitialPopulation(self):
        x1_min, x2_min = 0, 0
        x1_max, x2_max = self.database_size - 1, 1
        self.population = [(random.randint(x1_min, x1_max), random.random()) for _ in range(self.population_size)]

    def computeK(self):
        pass

    def calculateFitnessFunction(self):
        pass

    def calculateSwarmBest(self):
        pass

    def calculateVelocity(self):
        pass

    def calculatePosition(self):
        pass

    def algorithmLoop(self):
        self.normalization()
        print(self.data.head())
