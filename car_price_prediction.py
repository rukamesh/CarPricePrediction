from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from utils import *
from pyspark import SparkContext, SparkConf

file_path = "./data.csv"
checkpoint_dir = "./CheckpointDir/"
conf = SparkConf().setAppName("Car Price Prediction").setMaster("local[*]")
sc = SparkContext(conf=conf)
print(sc.getConf().getAll())
sc.setCheckpointDir(checkpoint_dir)
spark = SQLContext(sc)

data = spark.read.csv(path=file_path, header=True, quote='"', sep=",", inferSchema=True)
data_test, data_train = data.randomSplit(weights=[0.3, 0.7], seed=10)

get_indexer_input = get_indexer_input(data)


def model_training(data_train, indexer_input):
    x_cols = list(set(data_train.columns) - set(indexer_input.keys() + ["Price"]))
    str_ind_cols = ['indexed_' + column for column in indexer_input.keys()]
    indexers = indexer_input.values()
    pipeline_tr = Pipeline(stages=indexers)
    data_tr = pipeline_tr.fit(data_train).transform(data_train)
    assembler = VectorAssembler(inputCols=x_cols, outputCol="features")
    gbt = GBTRegressor(featuresCol="features", labelCol="Price", stepSize=0.008, maxDepth=5, subsamplingRate=0.75,
                       seed=10, maxIter=500, minInstancesPerNode=5, checkpointInterval=100, maxBins=64)
    pipeline_training = Pipeline(stages=[assembler, gbt])
    model = pipeline_training.fit(data_tr)
    return model


def model_testing(model, data_test, indexer_input):
    indexers = indexer_input.values()
    pipeline_te = Pipeline(stages=indexers)
    data_te = pipeline_te.fit(data_test).transform(data_test)
    predictions = model.transform(data_te)
    predictions.select("Price", "Mileage", "Make", "Model", "Trim", "Type", "prediction").toPandas().to_csv(
        './Output/prediction_without_categorical_variable.csv')
    return "model testing file saved"


model = model_training(data_train, get_indexer_input)
model_testing(model, data_test, get_indexer_input)
