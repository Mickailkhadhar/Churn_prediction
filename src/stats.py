import numpy as np
class Stats:
    def __init__(self):
        pass

# apply min max normalization of all the column needed
    def normalize(self, data, cols_to_norm):
        try:
            data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        except Exception as e:
            print(f'Error : {str(e)}')
        return data

#return max of column
    def get_max(self, data, column):
        return max(data[column])

    #return min of column
    def get_min(self, data, column):
        return min(data[column])

    #return variance of column
    def get_variance(self, data, column):
        return np.var(data[column])
