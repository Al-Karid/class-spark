import os
import pandas as pd

def load(fpath, s=","):
    flist = dict()
    for f in os.listdir(fpath):
        flist[f] = pd.read_csv(fpath+f, sep=s)
    return flist