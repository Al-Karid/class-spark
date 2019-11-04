import numpy as np
import pandas as pd

def impute_missing_values(df):
    for col in df.columns:
        if df[col].dtype == np.float:
            df[col] = df[col].fillna(df[col].median())
        if df[col].dtype == np.object:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

# Age categorical split
def is_child(v, age=10):
    v = np.where(v<age,"yes","no")
    return pd.Series(v, name="Age")

# Create matrix by adding columns with same height
def create_matrix(features, make_dummies=[]):
    variables = []
    for idx, f in enumerate(features):
        if make_dummies[idx]==True:
            variables.append(pd.get_dummies(f, prefix=f.name))
        else: variables.append(f)
    tmp_df = pd.concat(variables, axis=1)
    return impute_missing_values(tmp_df)

# Get titles
def get_passager_title(p_names):
    titles = []
    for idx, n in p_names.iteritems():
        titles.append(n.split(",")[1].split(".")[0].strip())
    return pd.Series(titles, name="Name")

# Check surnames
def has_surname(p_names):
    return p_names.apply(lambda x: "(" in x)

def clean_testset(X, XtestX):
    d = dict()
    for c in X.columns:
        if c not in XtestX.columns:
            d[c] = np.repeat(0,XtestX.shape[0])
    XtestXX = XtestX.join(pd.DataFrame(d))

    for c in XtestXX.columns:
        if c not in X.columns:
            XtestXX = XtestXX.drop(c, axis=1)
    return XtestXX