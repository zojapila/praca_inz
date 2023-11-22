from typing import Tuple

import pandas as pd


class DataPreprocessing:
    def __init__(self, path: str) -> None:
        self.original_df: pd.DataFrame = None
        self.unprocessed_dataframe: pd.DataFrame = None
        self.processed_dataframe: pd.DataFrame = None

        self.min_max_column_vals: dict = {}
        self.conversion_dicts: dict = {}
        self.column_labels: dict = {}

        self.loadDatabase(path)
        # used_features = ['id', 'srcip', "sport", 'dstip', 'dsport', 'sbytes', 'dbytes', 'state', 'dur', 'proto', 
        #                  'service', 'trans_depth', 'attack_cat', 'label']
        self.removeUselessColumns(['sbytes', 'dbytes', 'is_ftp_login',
                                   'swin', 'dwin', 'trans_depth'])
        self.selectFeatures()
        self.convertStringtoInt(True)
        self.removeStringColumns()
        self.findMinAndMaxValues()
        self.attack_df, self.normal_df = self.makeDfsForAttacksAndNormalEntries()
        self.column_labels = self.getColumnLabels()

    '''
    @brief: load database from file
    '''
    def loadDatabase(self, path: str) -> bool:
        # for kdd database
        # cols="""duration, protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment,
        #         urgent, hot, num_failed_logins,
        #         logged_in, num_compromised, root_shell, su_attempted, num_root, num_file_creations,
        #         num_shells, num_access_files,
        #         num_outbound_cmds, is_host_login, is_guest_login, count, srv_count, serror_rate,
        #         srv_serror_rate, rerror_rate,
        #         srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_count, dst_host_srv_count,
        #         dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate,
        #         dst_host_srv_diff_host_rate,
        #         dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate"""

        # columns=[]
        # for c in cols.split(','):
        #     if(c.strip()):
        #         columns.append(c.strip())

        # columns.append('target')

        # self.unprocessed_dataframe = pd.read_csv(path,names=columns)

        # for unsw_nb15
        self.original_df = pd.read_csv(path)
        self.unprocessed_dataframe = pd.read_csv(path)
        # for testing purposes
        # self.unprocessed_dataframe = self.unprocessed_dataframe.head(5000)
        return True

    '''
    @brief: converts data columns which datatype is string to float using 1-from-n method. Function creates 
            conversion dict for easy access to the converted data
    '''
    def convertStringtoInt(self, all_cols: bool = True, column_name: str = None) -> bool:
        if not all_cols:
            new_column_name = column_name + "_converted"
            conversion_vals = self.processed_dataframe[column_name].unique()
            # dictionary converting string to int
            conversion_dict = {string: numer for numer, string in enumerate(conversion_vals)}
            # adding new column 
            self.processed_dataframe[new_column_name] = self.processed_dataframe[column_name].map(conversion_dict)
            self.conversion_dicts[new_column_name] = conversion_dict
        else:
            # print(self.processed_dataframe.head())
            for column_name, dtype in self.processed_dataframe.dtypes.items():
                # print(column_name)
                if dtype not in ['int64', 'float64'] and column_name != "attack_cat":
                    new_column_name = column_name + "_converted"
                    conversion_vals = self.processed_dataframe[column_name].unique()
                    # dictionary converting string to int
                    conversion_dict = {string: numer for numer, string in enumerate(conversion_vals)}
                    # adding new column 
                    self.processed_dataframe[new_column_name] = \
                        self.processed_dataframe[column_name].map(conversion_dict)
                    self.conversion_dicts[new_column_name] = conversion_dict
            # print(self.processed_dataframe.head())

        return True

    '''
    @brief: adds min and max value and datatype for all of the columns to the dict
    '''

    def findMinAndMaxValues(self) -> bool:
        for column, dtype in self.unprocessed_dataframe.dtypes.items():
            if dtype in ['int64', 'float64'] and not column == "label" and not column == 'id':
                self.min_max_column_vals[column] = [self.unprocessed_dataframe[column].min(), 
                                                    self.unprocessed_dataframe[column].max(), 
                                                    dtype]
        return True

    '''
    @brief: if name od the features to be selected from the unprocessed data given it creates a processed dataframe
            using only those, if else, just unprocessed data is copied to processed_dataframe variable
    '''
    def selectFeatures(self, features: list = None):
        if features:
            self.processed_dataframe = self.unprocessed_dataframe[features]
        else:
            self.processed_dataframe = self.unprocessed_dataframe
        return True

    '''
    @brief: removes the columns which datatype is string
    '''
    def removeStringColumns(self) -> bool:
        for column, dtype in self.processed_dataframe.dtypes.items():
            if dtype not in ['int64', 'float64']:
                self.processed_dataframe.drop(column, axis='columns', inplace=True)
        return True

    '''
    @brief: creates dataframes with only attack or normal data entries
    '''
    def makeDfsForAttacksAndNormalEntries(self) -> Tuple:
        attack_df = self.processed_dataframe[self.processed_dataframe['label'] == 1]
        attack_df = attack_df.drop(columns=['label'])
        normal_df = self.processed_dataframe[self.processed_dataframe['label'] == 0]
        normal_df = normal_df.drop(columns=['label'])

        return attack_df, normal_df

    '''
    @brief: returns list of the column labels in processed dataframe
    '''
    def getColumnLabels(self) -> list:
        labels = self.processed_dataframe.columns.tolist()
        return labels

    '''
    brief: removes data that should be ignored from the unprocessed dataframe
    '''
    def removeUselessColumns(self, names: list) -> bool:
        self.unprocessed_dataframe = self.unprocessed_dataframe.drop(columns=names)
        # print(self.unprocessed_dataframe.head())
        return True





