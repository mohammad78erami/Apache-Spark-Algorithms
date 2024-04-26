#!pip install pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("/iris.csv", inferSchema =True, header=True)

features =  ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
vec_assembler = VectorAssembler(inputCols = features,
                                outputCol = "features")
data = vec_assembler.transform(df)

labelIndexer = StringIndexer(inputCol="class", outputCol="indexedLabel").fit(data)

featureIndexer =\
     VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

(trainingData, testData) = data.randomSplit([0.7, 0.3])

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

model = pipeline.fit(trainingData)

predictions = model.transform(testData)

predictions.select("predictedLabel", "class", "features").show(100)

evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
print("accuracy is ", accuracy)

rfModel = model.stages[2]
print(rfModel)  # summary only
