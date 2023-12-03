import pandas as pd
import data_preperation.data_preprocessing as preprocessing

training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"

processed_data = preprocessing.DataPreprocessing(training_data)
values = processed_data.attack_df.apply(lambda x: x.value_counts())
for i in processed_data.column_labels:
    if i != 'label':
        print(processed_data.normal_df[i].value_counts(normalize=True))
        znak = input("Wprowadź cokolwiek i naciśnij Enter: ")


# print(values)
# print(processed_data.attack_df.value_counts())
# processed_data.normal_df()
