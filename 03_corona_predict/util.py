import numpy as np

def create_data_set(data_set, look_back):
    x_data = []
    y_data = []
    for i in range(len(data_set) - look_back):
        data = data_set[i:(i + look_back), 0]
        x_data.append(data)
        y_data.append(data_set[i + look_back, 0])
    return np.array(x_data), np.array(y_data)
