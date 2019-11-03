from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from functions.data import split_data

def get_eval(model, tdf, mName="areaUnderROC"):
    
    # Evaluating the model
    pred = model.evaluate(tdf).predictions

    # Columns
    tlabels = list(pred.columns)[1]

    # Confusion matrix components
    tp = pred[(pred.flg_cmd_lowcostIndex == 1.0) & (pred.prediction == 1.0)].count()
    tn = pred[(pred.flg_cmd_lowcostIndex == 0.0) & (pred.prediction == 0.0)].count()

    # Metrics
    accuracy = float((tp+tn) /(pred.count()))
    auc = BinaryClassificationEvaluator(labelCol=tlabels, metricName=mName).evaluate(pred)
    
    # Return
    return accuracy, auc

def eval_lr(df, split=[0.4,0.6], enetparam=0.5, mName="areaUnderROC"):
    
    # Spliting the data
    train_df, test_df = split_data(df, split=split)
    
    # Training the model
    lr = LogisticRegression(labelCol=train_df.columns[1], featuresCol=train_df.columns[0],elasticNetParam=enetparam)
    model_lr = lr.fit(train_df)

    return get_eval(model_lr, test_df)

def eval_rf(df, split=[0.4,0.6], enetparam=0.5, mName="areaUnderROC"):
    
    # Spliting the data
    train_df, test_df = split_data(df, split=split)

    # Training the model
    model_rf = RandomForestClassifier(labelCol="flg_cmd_lowcostIndex", featuresCol="indexedFeatures",
                                    maxDepth=15, numTrees=100)
    model_rf = model_rf.fit(train_df)

    # return
    return get_eval(model_rf, test_df)