import random
import numpy as np
import pandas as pd
from data_preperation.data_preprocessing import DataPreprocessing
from sklearn.preprocessing import MinMaxScaler


class ParticleSwarmOptimization:
    def __init__(self, data: DataPreprocessing, population_size: int = 100, max_iter: int = 100,
                 final_sol_num: int = 10):
        self.population_size = population_size
        # self.data_unedited = data.processed_dataframe
        self.data = data.processed_dataframe
        self.data = self.data.drop(labels=["id"], axis=1)
        self.unlabeled_data = None
        self.database_size = self.data.shape[0]
        self.x_i = []
        self.v_i = []
        self.k_table = [0 for _ in range(population_size)]
        self.fitness_table = [0 for _ in range(population_size)]
        # self.x_table = []
        # self.v_table = []
        self.x_best_table = []
        self.x_global_best = None
        self.max_iter = max_iter
        self.final_sol_num = final_sol_num
        self.final_solution = []

    def normalization(self):
        scaler = MinMaxScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        self.unlabeled_data = self.data.drop(labels=['label'], axis=1)
        self.unlabeled_data.to_csv('pso_df.csv', index=False)

    # population as if (ID, r)
    '''
    @brief: generate velocities and positions for initial population
    x_i are (ID, r)
    v_i are (v ID and v r)
    '''
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

    # TODO: FIGURE OUT THE COMPUTING AFTER HAVING THE DATABASES
    def computeK(self, point_id: int, r: float) -> int:
        k = 0
        for index, row in self.unlabeled_data.iterrows():
            if index != point_id:
                diff = np.subtract(self.unlabeled_data.iloc[point_id], row)
                dist = np.sum(np.power(diff, 2))
                if dist <= r ** 2:
                    k += 1
        return k

    '''
    @brief: function to generate databases with the distances between points
    '''
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

    '''
    @brief: calculate fitness function for the element with given index
    '''
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

    '''
    @brief: update personal best position and fitness table of the given index
    '''
    def updatePersonalBest(self, index: int):
        if self.fitness_table[index] > self.x_best_table[index][1]:
            self.x_best_table[index] = (self.x_i[index], self.fitness_table[index])

    '''
    @brief: calculate velocities and positions of every element of the population
    '''
    def calculateVelocityAndPosition(self):
        phi1 = 1
        phi2 = 1
        c1 = 1
        c2 = 1
        for i in range(self.population_size):
            self.v_i[i] = (0.729 * (self.v_i[i][0] + phi1 * c1 * (self.x_best_table[i][0][0] - self.x_i[i][0]) +
                                    phi2 * c2 * (self.x_global_best[0][0] - self.x_i[i][0])),
                           0.729 * (self.v_i[i][1] + phi1 * c1 * (self.x_best_table[i][0][1] - self.x_i[i][1]) +
                                    phi2 * c2 * (self.x_global_best[0][1] - self.x_i[i][1])))
            self. x_i[i] = (round(self.x_i[i][0] + self.v_i[i][0]),
                            self.x_i[i][1] + self.v_i[i][1] if self.x_i[i][1] + self.v_i[i][1] <= 1 else 1)


    '''
    @brief: getter for the element with the best position
    '''
    def getSwarmBest(self):
        self.x_global_best = max(self.x_best_table, key=lambda x: x[1])

    def getFinalSolution(self):
        print('x_table: ', self.x_best_table)
        result_data = sorted(self.x_best_table, key=lambda x: x[1], reverse=True)[:self.final_sol_num]

        # result_data = sorted(range(len(self.x_table)), key=lambda x: x[1], reverse=True)[:self.final_sol_num]
        print('result data', result_data)
        # wyżej zapis (ID, r), fit, chciałabym df z kolumną r
        df1 = pd.DataFrame([i[0] for i in result_data], columns=['id', 'r'])
        # print('result ids and rs', df1.head())
        result_df = pd.merge(self.data, df1, left_index=True, right_on='id', how='inner')
        result_df.to_csv("pso_results.csv", index=False)
        return result_df

    def algorithmLoop(self):
        # generate initial population
        self.normalization()
        print('data:')
        print(self.data.head())
        self.generateInitialPopulation()
        # self.saveToCsvComputing()
        for _ in range(self.max_iter - 1):
            for i in range(self.population_size):
                self.k_table[i] = self.computeK(self.x_i[i][0], self.x_i[i][1])
                self.fitness_table[i] = self.calculateFitnessFunction(i)
                self.updatePersonalBest(i)
                # print(i)
            # print(self.fitness_table)
            self.getSwarmBest()
            # print(self.x_global_best)
            self.calculateVelocityAndPosition()
            print(self.x_i)
            # print(self.v_i)
        # last element
        for i in range(self.population_size):
            self.k_table[i] = self.computeK(self.x_i[i][0], self.x_i[i][1])
            self.fitness_table[i] = self.calculateFitnessFunction(i)
            self.updatePersonalBest(i)
            print(i)

        print("fit table", self.fitness_table)
        self.getSwarmBest()

        # print(self.x_global_best)
        self.final_solution = self.getFinalSolution()
        print(self.final_solution.head())
