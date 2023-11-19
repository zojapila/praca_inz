import random
import numpy as np
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
        self.k_table = [0 for _ in range(population_size)]
        self.fitness_table = [0 for _ in range(population_size)]
        self.x_table = []
        self.v_table = []
        self.x_best_table = []

    def normalization(self):
        scaler = MinMaxScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        self.unlabeled_data = self.data.drop(labels=['label'], axis=1)

    # population as if (ID, r)
    def generateInitialPopulation(self):
        x1_min, x2_min = 0, 0
        x1_max, x2_max = self.database_size - 1, 1
        self.population = [(random.randint(x1_min, x1_max), random.random()) for _ in range(self.population_size)]

    def computeK(self, point_id: int, r: float) -> int:
        k = 0
        for index, row in self.unlabeled_data.iterrows():
            if index != point_id:
                diff = np.subtract(self.unlabeled_data.iloc[point_id], row)
                dist = np.sum(np.power(diff, 2))
                if dist <= r ** 2:
                    k += 1
        return k

    def saveToCsvComputing(self):
        data = {}
        for i in range(self.database_size):
            if i % 10 != 0:
                data[i] = []
                for index, row in self.unlabeled_data.iterrows():
                    diff = np.subtract(self.unlabeled_data.iloc[i], row)
                    dist = np.sum(np.power(diff, 2))
                    data[i].append(dist)
            else:
                df = pd.DataFrame(data)
                df.to_csv(f'datafile{i}.csv', index=False)
                data = {}

            print(i)




    def calculateFitnessFunction(self, index: int) -> float:
        alpha = 0.05 * self.database_size
        fit = (((alpha/(self.population[index][1] * self.k_table[index])) +
               (self.k_table[index]/self.population[index][1])) +
               (self.k_table[index]/(self.database_size - self.k_table[index])))
        return fit

    def calculateSwarmBest(self):
        pass

    def calculateVelocity(self):
        pass

    def calculatePosition(self):
        pass

    def algorithmLoop(self):
        self.normalization()
        print(self.data.head())
        self.generateInitialPopulation()
        self.saveToCsvComputing()
        # for i in range(len(self.population)):
        #     self.k_table[i] = self.computeK(self.population[i][0], self.population[i][1])
            # self.fitness_table[i] = self.calculateFitnessFunction(i)

            # print(self.k_table[i], self.fitness_table[i])
