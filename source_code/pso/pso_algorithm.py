import random
import numpy as np
import pandas as pd
from data_preperation.data_preprocessing import DataPreprocessing
from sklearn.preprocessing import MinMaxScaler


class ParticleSwarmOptimization:
    def __init__(self, data: DataPreprocessing, population_size: int = 100, max_iter = 100):
        self.population_size = population_size
        self.data = data.processed_dataframe
        self.data = self.data.drop(labels=["id"], axis=1)
        self.unlabeled_data = None
        self.database_size = self.data.shape[0]
        self.x_i = []
        self.v_i = []
        self.k_table = [0 for _ in range(population_size)]
        self.fitness_table = [0 for _ in range(population_size)]
        self.x_table = []
        self.v_table = []
        self.x_best_table = []
        self.x_global_best = None
        self.max_iter = max_iter

    def normalization(self):
        scaler = MinMaxScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        self.unlabeled_data = self.data.drop(labels=['label'], axis=1)

    # population as if (ID, r)
    def generateInitialPopulation(self):
        x1_min, x2_min = 0, 0
        x1_max, x2_max = self.database_size - 1, 1
        v1_min = -10
        v1_max = 10
        v2_min = -1
        v2_max = 1
        self.x_i = [(random.randint(x1_min, x1_max), random.random()) for _ in range(self.population_size)]
        self.v_i = [(random.uniform(v1_min, v1_max), random.uniform(v2_min, v2_max))
                    for _ in range(self.population_size)]
        for i in self.x_i:
            self.x_best_table.append((i, 0))

    def computeK(self, point_id: int, r: float) -> int:
        # todo: no i co tera
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
        for i in range(291, 300):
            if i % 10 != 0:
                data[i] = []
                for index, row in self.unlabeled_data.iterrows():
                    diff = np.subtract(self.unlabeled_data.iloc[i], row)
                    dist = np.sum(np.power(diff, 2))
                    data[i].append(dist)
                print(i)
            else:
                data[i] = []
                for index, row in self.unlabeled_data.iterrows():
                    diff = np.subtract(self.unlabeled_data.iloc[i], row)
                    dist = np.sum(np.power(diff, 2))
                    data[i].append(dist)
                print(i)
                df = pd.DataFrame(data)
                df.to_csv(f'datafile{i}.csv', index=False)
                data = {}

    def calculateFitnessFunction(self, index: int) -> float:
        alpha = 0.05 * self.database_size
        if self.x_i[index][1] != 0 and self.k_table[index] != 0:
            fit = (((alpha / (self.x_i[index][1] * self.k_table[index])) +
                    (self.k_table[index] / self.x_i[index][1])) +
                   (self.k_table[index]/(self.database_size - self.k_table[index])))
        elif self.x_i[index][1] != 0 and self.k_table[index] == 0:
            fit = ((self.k_table[index] / self.x_i[index][1]) +
                   (self.k_table[index] / (self.database_size - self.k_table[index])))
        else:
            fit = self.k_table[index]/(self.database_size - self.k_table[index])
        return fit

    def updatePersonalBest(self, index: int):
        if self.fitness_table[index] > self.x_best_table[index][1]:
            self.x_best_table[index] = (self.x_i[index], self.fitness_table[index])

    def calculateVelocityAndPosition(self):
        phi1 = 1
        phi2 = 1
        c1 = 1
        c2 = 1
        # todo: jak zrobic zeby nie byl r wiekszy od 1
        for i in range(self.population_size):
            self.v_i[i] = (0.729 * (self.v_i[i][0] + phi1 * c1 * (self.x_best_table[i][0][0] - self.x_i[i][0]) +
                                    phi2 * c2 * (self.x_global_best[0][0] - self.x_i[i][0])),
                           0.729 * (self.v_i[i][1] + phi1 * c1 * (self.x_best_table[i][0][1] - self.x_i[i][1]) +
                                    phi2 * c2 * (self.x_global_best[0][1] - self.x_i[i][1])))
            self. x_i[i] = (round(self.x_i[i][0] + self.v_i[i][0]), self.x_i[i][1] + self.v_i[i][1])

    def getSwarmBest(self):
        self.x_global_best = max(self.x_best_table, key=lambda x: x[1])

    def algorithmLoop(self):
        self.normalization()
        print(self.data.head())
        self.generateInitialPopulation()
        # self.saveToCsvComputing()
        for i in range(self.max_iter):
            for i in range(self.population_size):
                self.k_table[i] = self.computeK(self.x_i[i][0], self.x_i[i][1])
                self.fitness_table[i] = self.calculateFitnessFunction(i)
                self.updatePersonalBest(i)
                print(i)
            print(self.fitness_table)
            self.getSwarmBest()
            print(self.x_global_best)
            self.calculateVelocityAndPosition()
            print(self.x_i)
            print(self.v_i)
        print(self.x_global_best)
