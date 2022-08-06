from typing import List, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from encoder import CategoricalToIntegerEncoders


class GetXY:
    def __init__(
        self,
        spark: SparkSession,
        feature_columns: List[str],
        categorical_columns: List[str],
        target_column: str
    ):
        self.spark = spark
        self.target_column = target_column

        feature_columns = set(feature_columns)
        categorical_columns = set(categorical_columns)

        categorical_not_in_feature = categorical_columns - feature_columns
        non_categorical_feature_columns = feature_columns - categorical_columns

        if len(categorical_not_in_feature) > 0:
            raise ValueError(
                f'Categorical columns: {categorical_not_in_feature} not found in feature_columns'
            )
        self.categorical_columns = list(categorical_columns)
        self.non_categorical_feature_columns = list(
            non_categorical_feature_columns)

    def fit(self, df: DataFrame):
        self.categorical_encoders = (
            CategoricalToIntegerEncoders(columns=self.categorical_columns)
            .fit(df)
        )
        return self

    @property
    def columns_encoded(self) -> List[str]:
        return self.categorical_encoders.columns_encoded

    def transform(self, df: DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.categorical_encoders.transform(spark=self.spark, df=df)

        has_target = self.target_column in df.columns
        select_cols = self.non_categorical_feature_columns + self.columns_encoded
        if has_target:
            select_cols.append(self.target_column)

        pdf = df.select(select_cols).toPandas()
        return (
            pdf[self.non_categorical_feature_columns + self.columns_encoded],
            pdf[self.target_column] if has_target else None
        )
