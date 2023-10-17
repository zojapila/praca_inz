import pandas as pd

class dataPreprocessing():
    def __init__(self) -> None:
        self.unprocessed_dataframe : pd.DataFrame = None
        self.processed_dataframe : pd.DataFrame = None
        self.min_max_column_vals : dict = {}
        
    def loadDatabase(self, path: str) -> bool:
        cols="""duration, protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment, urgent, hot, num_failed_logins,
                logged_in, num_compromised, root_shell, su_attempted, num_root, num_file_creations, num_shells, num_access_files,
                num_outbound_cmds, is_host_login, is_guest_login, count, srv_count, serror_rate, srv_serror_rate, rerror_rate,
                srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_count, dst_host_srv_count,
                dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate,
                dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate"""

        columns=[]
        for c in cols.split(','):
            if(c.strip()):
                columns.append(c.strip())

        columns.append('target')

        self.unprocessed_dataframe = pd.read_csv(path,names=columns)
        return True


    def convertStringtoInt(self):
        pass


    def findMinAndMaxValues(self):
        for column, dtype in self.unprocessed_dataframe.dtypes.items():
            if dtype in ['int64', 'float64']:
                self.min_max_column_vals[column] = [self.unprocessed_dataframe[column].min(), self.unprocessed_dataframe[column].max(), dtype]
        return True



training_data = "D:/studia/inzynierka/data/kddcup.data_10_percent.gz"
data = dataPreprocessing()
data.loadDatabase(training_data)
data.findMinAndMaxValues()
