#!pip install pyspark
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import when

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

train = trainingData.withColumnRenamed("class", "label")
test = testData.withColumnRenamed("class", "label")

training = train.withColumn("label", when(train.label == "Iris-setosa","0") \
      .when(train.label == "Iris-versicolor","1")\
      .when(train.label == "Iris-virginica", "2")\
      .otherwise(train.label))
training = training.withColumn("label", training.label.cast("int"))

test_data = test.withColumn("label", when(test.label == "Iris-setosa","0") \
      .when(test.label == "Iris-versicolor","1")\
      .when(test.label == "Iris-virginica", "2")\
      .otherwise(test.label))
test_data = test_data.withColumn("label", test_data.label.cast("int"))


# train the model
model = nb.fit(training)

# select example rows to display.
predictions = model.transform(test_data)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                              predictionCol="prediction",
                                              metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
