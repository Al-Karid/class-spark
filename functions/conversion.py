# The function

def cast_columns_of_df(df, cols_to_cast, col_to_keep, cast_type='double'):
    """cast continuous columns into double since all columns are """
    return df.select(col_to_keep + [(df[feature].cast('double')) for feature in cols_to_cast if 'ID_CLIENT' not in feature])