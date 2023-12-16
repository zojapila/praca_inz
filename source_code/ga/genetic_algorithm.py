import random

import pandas as pd

# from numpy import idxmax

from data_preperation.data_preprocessing import DataPreprocessing
# import pandas as pd
# import numpy as np


class GeneticAlgorithm:
    def __init__(self, data: DataPreprocessing, initial_population_size: int = 100, max_iter: int = 100,
                 mutation_probability: float = 0.4, num_of_final_sol=20, filenum: int = 1) -> None:
        self.population_size: int = initial_population_size
        self.data: DataPreprocessing = data
        self.max_iter = max_iter

        self.column_list = data.column_labels
        self.column_list.remove('id')
        self.column_list.remove('label')
        self.weights = []
        self.mutation_probability = mutation_probability

        self.population: dict = {}
        self.evaluation_results: list = []
        self.first_free_idx = 0
        self.num_of_final_sol = num_of_final_sol
        self.filenum = filenum
        # self.generateInitialPopulation()
        # self.evaluationFunction(1)

        # print(self.population)

    '''
    @brief: generates initial population using chosen method:
    - version 1: random elements in previously calculated ranges
    - version 2: random element from the given database is chosen
    '''
    def generateInitialPopulation(self, pop_type: int = 1) -> bool:
        if pop_type == 1:
            # version 1 - create population with random elements in previously calculated ranges
            for idx in range(0, self.population_size):
                chromosome = []
                for label, vals in self.data.min_max_column_vals.items():
                    if vals[2] == 'float64':
                        chromosome.append(random.uniform(vals[0], vals[1]))
                    else:
                        chromosome.append(random.randint(vals[0], vals[1]))
                self.population[idx] = chromosome
                # print(chromosome)
        else:
            # version 2 - take random elements from the given dataframe
            for idx in range(0, self.population_size):
                chromosome = []
                num = random.randint(0, self.data.processed_dataframe.shape[0] - 1)
                for col_name, val in self.data.processed_dataframe.iloc[num].items():
                    if col_name not in ["id", "label"]:
                        chromosome.append(val)
                self.population[idx] = chromosome
        self.first_free_idx = self.population_size
        # print(len(self.population[0]))
        return True

    '''
    @brief: evaluation function for given element od the population is being calculated 
    '''
    def evaluationFunction(self, idx) -> float:
        idx_appears = []
        quantity = [0 for _ in range(self.getChromosomeLength())]

        for i in self.data.attack_df.itertuples():
            for column in range(0, self.getChromosomeLength()):
                if i[column + 2] == self.population[idx][column]:
                    if column not in idx_appears:
                        idx_appears.append(column)
                    # print(self.data.column_labels[column])
                    quantity[column] += 1

                    # result += weights[column]
                    # indexes_to_check.remove(column)
        result = sum([i * self.weights[i] / self.data.processed_dataframe.shape[0] for i in idx_appears])
        ranking = 1  # temporary solution
        x = sum(quantity)
        # x += [i for i in quantity]
        result = x - (result * ranking) / 100
        # print(self.column_list[quantity.index(max(quantity))])
        return result

    def generateWeights(self, rand: bool = False) -> list:
        # weights = []
        k = self.getChromosomeLength()
        # print(k)
        if rand:
            weights = random.sample(population=[_ for _ in range(1, k + 1)], k=k)
            return weights
        else:
            weights = [3, 3, 3, 3, 2, 2, 3, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 3, 3, 3]
        return weights

    def mutation(self, idx: int):
        elem_to_be_mutated = random.randint(0, len(self.population[idx]) - 1)
        col_name = self.data.column_labels[elem_to_be_mutated]
        if self.data.min_max_column_vals[col_name][-1] == "float64":
            self.population[idx][elem_to_be_mutated] = random.uniform(self.data.min_max_column_vals[col_name][0],
                                                                      self.data.min_max_column_vals[col_name][1])
        else:
            self.population[idx][elem_to_be_mutated] = random.randint(self.data.min_max_column_vals[col_name][0],
                                                                      self.data.min_max_column_vals[col_name][1])
        return True

    def selection(self):
        self.evaluation_results = sorted(self.evaluation_results, key=lambda x: x[1],
                                         reverse=True)[:int(self.population_size*0.6)]
        # print(self.evaluation_results)
        pairs_to_be_crossed = []
        for _ in range(self.population_size - len(self.evaluation_results)):
            p1 = p2 = self.evaluation_results[random.randint(0, len(self.evaluation_results)-1)][0]
            while p1 == p2:
                p2 = self.evaluation_results[random.randint(0, len(self.evaluation_results)-1)][0]
            pairs_to_be_crossed.append((p1, p2))

        # pairs_to_be_crossed = [(self.evaluation_results[random.randint(0, len(self.evaluation_results)-1)][0],
        #                         self.evaluation_results[random.randint(0, len(self.evaluation_results)-1)][0])
        #                        for _ in range(self.population_size - len(self.evaluation_results))]
        # (random.randint(0, n-1), random.randint(0, n-1)) for _ in range(x)
        for pairs in pairs_to_be_crossed:
            self.crossing(pairs[0], pairs[1])

        return True

    def crossing(self, id1: int, id2: int) -> bool:
        point_of_crossing = [random.randint(2, self.getChromosomeLength() - 3) for _ in range(3)]
        while point_of_crossing[1] == point_of_crossing[0]:
            point_of_crossing[1] = random.randint(2, self.getChromosomeLength() - 3)
        while point_of_crossing[2] == point_of_crossing[0] or point_of_crossing[1] == point_of_crossing[2]:
            point_of_crossing[2] = random.randint(2, self.getChromosomeLength() - 3)
        point_of_crossing = sorted(point_of_crossing)
        child1 = [self.population[id1][i] for i in range(0, point_of_crossing[0])]
        for i in range(point_of_crossing[0], point_of_crossing[1]):
            child1.append(self.population[id2][i])
        for i in range(point_of_crossing[1], point_of_crossing[2]):
            child1.append(self.population[id1][i])
        for i in range(point_of_crossing[2], len(self.population[id2])):
            child1.append(self.population[id2][i])
        self.population[self.first_free_idx] = child1
        if random.random() <= self.mutation_probability:
            self.mutation(self.first_free_idx)
        self.evaluation_results.append((self.first_free_idx, self.evaluationFunction(self.first_free_idx)))
        self.first_free_idx += 1
        return True

    def getChromosomeLength(self) -> int:
        result = len(self.data.column_labels) - 2
        return result

    def getFinalSolution(self):
        solution_idxs = sorted(self.evaluation_results, key=lambda x: x[1], reverse=True)[:self.num_of_final_sol]
        solution = [self.population[idx] for idx, _ in solution_idxs]
        df_solution = pd.DataFrame(solution, columns=self.column_list)
        df_solution.to_csv('ga_results' + str(self.filenum) + '.csv', index=False)
        return df_solution

    def geneticAlgorithmLoop(self):
        # generate initial population
        self.generateInitialPopulation()
        self.weights = self.generateWeights(False)
        # evaluate initial population
        for keys, _ in self.population.items():
            self.evaluation_results.append((keys, self.evaluationFunction(keys)))
        # make selection
        for i in range(self.max_iter):
            self.selection()
            print(i)
            if i == 0:
                print(self.evaluation_results)
        print(sorted(self.evaluation_results, key=lambda x: x[1], reverse=True))
        return self.getFinalSolution()
        # for i in range(0, self.max_iter):
