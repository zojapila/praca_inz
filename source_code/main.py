import data_preperation.data_preprocessing as preprocessing
import ga.genetic_algorithm as ga


training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"
processed_data = preprocessing.DataPreprocessing(training_data)
print(processed_data.processed_dataframe.head())
# print(processed_data.column_labels)
# for i in processed_data.column_labels:
#     print(i)
# print(processed_data.min_max_column_vals)

genetic_algorithm = ga.GeneticAlgorithm(processed_data, 10)
genetic_algorithm.geneticAlgotrithmLoop()
print("dlugosc chromosomu:", len(genetic_algorithm.population[1]))

