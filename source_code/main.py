import data_preperation.data_preprocessing as preprocessing
import ga.genetic_algorithm as ga
import pso.pso_algorithm as pso


training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"
work_computer = "C:/praca_inz/source_code/praca_inz-master/source_code/UNSW_NB15_training-set.csv"
processed_data = preprocessing.DataPreprocessing(training_data)
# processed_data = preprocessing.DataPreprocessing(work_computer)

# ga
# genetic_algorithm = ga.GeneticAlgorithm(processed_data, initial_population_size=200, max_iter=10)
# genetic_algorithm.geneticAlgorithmLoop()

#pso
pso = pso.ParticleSwarmOptimization(processed_data)
pso.algorithmLoop()

