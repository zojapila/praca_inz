import data_preperation.data_preprocessing as preprocessing
import ga.genetic_algorithm as ga


training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"
work_computer = "C:/praca_inz/source_code/praca_inz-master/source_code/UNSW_NB15_training-set.csv"
processed_data = preprocessing.DataPreprocessing(training_data)
# processed_data = preprocessing.DataPreprocessing(work_computer)

# print(processed_data.processed_dataframe.info())
# print(processed_data.unprocessed_dataframe['label'].value_counts(normalize=True))
# print(processed_data.processed_dataframe['ct_flw_http_mthd'].value_counts(normalize=True))
# print(processed_data.attack_df['ct_flw_http_mthd'].value_counts(normalize=True))
# print(processed_data.normal_df.info())

genetic_algorithm = ga.GeneticAlgorithm(processed_data, 20)
# print(processed_data.processed_dataframe['is_sm_ips_ports'].value_counts())
# print(processed_data.processed_dataframe['ct_src_ltm'].value_counts())

genetic_algorithm.geneticAlgorithmLoop()
# print("chromosome length:", len(genetic_algorithm.population[1]))

