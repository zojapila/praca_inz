import data_preperation.data_preprocessing as preprocessing


training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"
processed_data = preprocessing.DataPreprocessing(training_data)
print(processed_data.processed_dataframe.head())


