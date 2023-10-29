import pandas as pd


class DataPreprocessing:
    def __init__(self, path: str) -> None:
        self.unprocessed_dataframe: pd.DataFrame = None
        self.processed_dataframe: pd.DataFrame = None
        self.min_max_column_vals: dict = {}
        self.conversion_dicts = {}

        self.loadDatabase(path)
        # used_features = ['id', 'srcip', "sport", 'dstip', 'dsport', 'sbytes', 'dbytes', 'state', 'dur', 'proto', 
        #                  'service', 'trans_depth', 'attack_cat', 'label']
        self.selectFeatures()
        # print(data.processed_dataframe.head())
        self.convertStringtoInt(True)
        self.findMinAndMaxValues()
        # print(data.conversion_dicts)
        # print(data.min_max_column_vals)
        self.removeStringColumns()

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
        self.unprocessed_dataframe = pd.read_csv(path)
        return True

    def convertStringtoInt(self, all: bool, column_name: str = None) -> bool:
        if column_name:
            new_column_name = column_name + "_converted"
            conversion_vals = self.processed_dataframe[column_name].unique()
            # dictionary converting string to int
            conversion_dict = {string: numer for numer, string in enumerate(conversion_vals)}
            # adding new column 
            self.processed_dataframe[new_column_name] = self.processed_dataframe[column_name].map(conversion_dict)
            self.conversion_dicts[new_column_name] = conversion_dict
        else:
            for column_name, dtype in self.processed_dataframe.dtypes.items():
                # print(column_name)
                if dtype not in ['int64', 'float64']:
                    new_column_name = column_name + "_converted"
                    conversion_vals = self.processed_dataframe[column_name].unique()
                    # dictionary converting string to int
                    conversion_dict = {string: numer for numer, string in enumerate(conversion_vals)}
                    # adding new column 
                    self.processed_dataframe[new_column_name] = \
                        self.processed_dataframe[column_name].map(conversion_dict)
                    self.conversion_dicts[new_column_name] = conversion_dict

        return True

    def findMinAndMaxValues(self) -> bool:
        for column, dtype in self.unprocessed_dataframe.dtypes.items():
            if dtype in ['int64', 'float64']:
                self.min_max_column_vals[column] = [self.unprocessed_dataframe[column].min(), 
                                                    self.unprocessed_dataframe[column].max(), 
                                                    dtype]
        return True

    def selectFeatures(self, features: list = None):
        if features:
            self.processed_dataframe = self.unprocessed_dataframe[features]
        else:
            self.processed_dataframe = self.unprocessed_dataframe
        return True
     
    def removeStringColumns(self) -> bool:
        for column, dtype in self.processed_dataframe.dtypes.items():
            if dtype not in ['int64', 'float64']:
                self.processed_dataframe.drop(column, axis = 'columns', inplace=True)
        return True



