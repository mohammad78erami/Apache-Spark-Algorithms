#!pip install pyspark

from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

Dataframe = spark.read.csv("/iris.csv", inferSchema=True, header=True)

features_col =  ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
vec_assembler = VectorAssembler(inputCols = features_col,
                                outputCol = "features")

data = vec_assembler.transform(Dataframe)
#split data to 80% train and 20% test
(train, test) = data.randomSplit([0.8,0.2])

Kmeans = KMeans(featuresCol = "features", k=3)
model = Kmeans.fit(train)
predictions = model.transform(test)

#showing the result in form of a table
model.transform(test).groupBy("prediction").count().show()
predictions.groupBy("class", "prediction").count().show()

#evaluation is done by using sillhoutte score
evaluator = ClusteringEvaluator()
sillhouette_score = evaluator.evaluate(predictions)
print("The Sillhouette score is: ", sillhouette_score)

#Showing result of KMeans Algorithm on a scatter plot
#since there are 4 features in input data(may be more depending on your data) and we need 2 dimentions for scatter plot, we gotta use PCA algorithm to reduce dimentions
from pyspark.ml.feature import PCA as PCAml
import numpy as np

pca = PCAml(k=2, inputCol="features", outputCol="pca")
pca_model = pca.fit(test)
pca_transformed = pca_model.transform(test)

x_pca = np.array(pca_transformed.rdd.map(lambda row: row.pca).collect())
cluster_assignment = np.array(predictions.rdd.map(lambda row: row.prediction).collect()).reshape(-1,1)
pca_data = np.hstack((x_pca,cluster_assignment))

#plotting the dimention reduced KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal","cluster_assignment"))
sns.FacetGrid(pca_df,hue="cluster_assignment", height=6).map(plt.scatter, '1st_principal', '2nd_principal' ).add_legend()

plt.show()
