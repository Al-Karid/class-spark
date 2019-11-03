import os

def load_data(ss, flocation):
    '''ss: spark session'''
    data_list = dict()
    for f in os.listdir(flocation):
        data_list[f] = ss.read.csv(flocation+f, header=True)
    return data_list

def join_data(dvector, col_to_keep):
    joined = dvector[0]
    for v in dvector[1:]:
        joined = joined.join(v, how="left", on=col_to_keep)
    return joined

def split_data(df, split=[0.4,0.6]):
    train_df, test_df = df.select(df.columns[0],df.columns[1]).randomSplit(split)
    return train_df, test_df