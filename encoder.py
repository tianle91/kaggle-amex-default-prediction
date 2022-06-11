from typing import Dict, List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType


class CategoricalToIntegerEncoder:
    def __init__(self, column: str):
        self.unique_values = None
        self.column = column
        self.column_encoded = f'{self.column}_CategoricalToIntegerEncoder'

    def fit(self, df: DataFrame):
        if self.unique_values is not None:
            raise ValueError('Already fit')
        self.unique_values = [
            row[self.column]
            for row in df.select(self.column).distinct().collect()
            if row[self.column] is not None
        ]
        return self

    def transform(self, spark: SparkSession, df: DataFrame):
        if self.unique_values is None:
            raise ValueError('Not fit yet')
        transformed_values = spark.createDataFrame(
            data=[
                {self.column: c, self.column_encoded: i}
                for i, c in enumerate(self.unique_values)
            ],
            schema=StructType(fields=[
                StructField(
                    self.column, dataType=StringType(), nullable=False),
                StructField(
                    self.column_encoded, dataType=IntegerType(), nullable=False),
            ])
        )
        return df.join(transformed_values, on=self.column, how='left').drop(self.column)


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
