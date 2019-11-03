# Low-cost customers

We achieved the scores below :

| Classifiers         | accuracy | AUC    |
| ------------------- | -------- | ------ |
| Logistic Classifier | 0.9656   | 0.9067 |
| Random Forest       | _        | _      |



### <u>Description</u>

All this project is done with **Spark**

This project is about **predicting low-cost customers** of a given transportation company.

For this purpose we created a module named **functions** with the following components as listed in the doc-string below.

### <u>Doc-string</u>

**<u>functions</u>**

​	**conversion**

> ​		cast_columns_of_df(df, cols_to_cast, cols_to_keep, cast_type) 
>

​			***return*** : same df with cols_to_cast in cast_type

​	**data**

> ​		load_data(ss, flocation)
>

​			ss : spark session

​			flocation : location of the data, in csv format. The path should only contain data to be loaded

​			***return*** : dictionary with loaded data name as key, the data as value

> ​		join_data(dvector, cols_to_keep)
>

​			dvector : vector of data to be joined

​			***return*** : one df of joined data from dvector

​	**features**

> ​		input_df(df) : removes missing values
>

> ​		preprocessed_df(df, label) : adds features vector to df
>

​			***return*** : df with additional features vector

​	**models**

> ​		eval_lr(df, split=[0.4,0.6], enetparam, mName)
>

​			df : df with only two columns, the features vector as the first and label as the second

​			split : how to split the data for cross-validation. Default value is [0.4,0.6]

​			enetparam : elasticnetParam of spark's logistic classifier

​			mName : the metric name to use

​			***return*** : accuracy, auc