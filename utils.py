from pyspark.sql.functions import col
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler


def get_indexer_input(data):
    str_cols_value = {}
    for c, t in data[data.columns].dtypes:
        if t == 'string':
            str_cols_value[c] = StringIndexer(inputCol=c, outputCol='indexed_' + c).fit(data)
    return str_cols_value


