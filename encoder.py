from typing import List

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession


class CategoricalToIntegerEncoder:
    def __init__(self, column: str):
        self.unique_values = None
        self.column = column
        self.column_encoded = f'{self.column}_CategoricalToIntegerEncoder'

    def fit(self, df: DataFrame):
        if self.unique_values is not None:
            raise ValueError('Already fit')
        self.unique_values = sorted([
            row[self.column]
            for row in df.select(self.column).distinct().collect()
            if row[self.column] is not None
        ])
        return self

    def transform(self, spark: SparkSession, df: DataFrame):
        if self.unique_values is None:
            raise ValueError('Not fit yet')
        mapping = {c: i for i, c in enumerate(self.unique_values)}
        udf = F.udf(lambda s: mapping.get(s), 'integer')
        return (
            df
            .withColumn(self.column_encoded, udf(self.column))
            .drop(self.column)
        )


class CategoricalToIntegerEncoders:
    def __init__(self, columns: List[str]) -> None:
        self.encoders = {
            c: CategoricalToIntegerEncoder(column=c)
            for c in columns
        }
        self.columns = columns
        self.columns_encoded = [
            self.encoders[c].column_encoded
            for c in columns
        ]

    def fit(self, df: DataFrame):
        for _, enc in self.encoders.items():
            enc.fit(df=df)
        return self

    def transform(self, spark: SparkSession, df: DataFrame):
        out = df
        for encoder in self.encoders.values():
            out = encoder.transform(spark=spark, df=out)
        return out
