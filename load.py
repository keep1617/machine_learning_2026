import numpy as np


def data_loader(path):

    loads = np.load(path)
    data = loads['arr_0']

    return data
