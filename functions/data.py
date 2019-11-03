import os

def load_data(ss, flocation):
    '''ss: spark session'''
    data_list = dict()
    for f in os.listdir(flocation):
        data_list[f] = ss.read.csv(f, header=True)
    return data_list

def join_data(dvector, col_to_keep):
    joined = dvector[0]
    for v in dvector[1:]:
        joined.join(v, how="left", on=col_to_keep)
    return joined