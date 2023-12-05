import pandas as pd
import data_preperation.data_preprocessing as preprocessing

training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"

processed_data = preprocessing.DataPreprocessing(training_data, 'ga')
# print(processed_data.column_labels)
x = ['id', 'spkts', 'dpkts', 'sttl', 'dttl', 'sloss', 'dloss', 'smean', 'dmean', 'response_body_len', 'ct_srv_src',
     'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd',
     'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'label', 'proto_converted',
     'service_converted', 'state_converted']
weights = [3, 3, 3, 3, 2, 2, 3, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 3, 3, 3]
# all values are ints
print(processed_data.min_max_column_vals)

# values = processed_data.normal_df.apply(lambda x: x.value_counts())
# for i in processed_data.column_labels:
#     if i != 'label':
#         print(processed_data.attack_df[i].value_counts(normalize=True))
#         znak = input("Wprowadź cokolwiek i naciśnij Enter: ")


# print(values)
# print(processed_data.attack_df.value_counts())
# processed_data.normal_df()
