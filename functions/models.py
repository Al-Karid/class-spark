from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def eval_lr(df, split=[0.4,0.6], enetparam=0.5, mName="areaUnderROC"):
    
    # Spliting the data
    train_df, test_df = df.select(df.columns[0],df.columns[1]).randomSplit(split)
    
    # Training the model
    lr = LogisticRegression(labelCol=df.columns[1], featuresCol=df.columns[0],elasticNetParam=enetparam)
    model_lr = lr.fit(train_df)

    # Evaluating the model
    pred_lr = model_lr.evaluate(test_df).predictions

    tp = pred_lr[(pred_lr.flg_cmd_lowcostIndex == 1.0) & (pred_lr.prediction == 1.0)].count()
    tn = pred_lr[(pred_lr.flg_cmd_lowcostIndex == 0.0) & (pred_lr.prediction == 0.0)].count()

    accuracy = float((tp+tn) /(pred_lr.count()))
    auc = BinaryClassificationEvaluator(labelCol=df.columns[1], metricName=mName).evaluate(pred_lr)

    return accuracy, auc