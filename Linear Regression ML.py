#!pip install pyspark
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# select example rows to display.
predictions = lrModel.transform(test_data)
predictions.show()

evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                              predictionCol="prediction",
                                              metricName="accuracy")

LRaccuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(LRaccuracy))
