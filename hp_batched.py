import json
from typing import List

import hyperopt
import mlflow
import pandas as pd
from pyspark.sql import DataFrame

from batched import BatchedLGBMClassifier
from evaluation import feval_amex, feval_amex_gini, feval_amex_top4
from getxy import GetXY

RANDOM_STATE = 20220731


def get_cv_hp_metrics(
    getxy: GetXY,
    train_data_labelled_batches: List[DataFrame],
    test_data_batches: List[DataFrame],
    lgb_params: dict,
    id_variables: List[str],
    prediction_variable: str,
) -> dict:
    with mlflow.start_run(nested=True) as run:
        mlflow.log_params(params=lgb_params)
        # auto logging converts everything to strings, we want something deserializable
        mlflow.log_param('lgb_params_json', json.dumps(lgb_params))

        model = BatchedLGBMClassifier(lgb_params=lgb_params, getxy=getxy).fit(
            dfs=train_data_labelled_batches)

        pred = model.predict(dfs=test_data_batches, id_variables=id_variables,
                             prediction_variable=prediction_variable)
        actual = pd.concat([
            df.select(*id_variables, getxy.target_column).toPandas()
            for df in test_data_batches
        ], axis=0)
        joined = pd.merge(left=actual, right=pred,
                          on=id_variables, how='inner')
        assert len(joined) == len(actual) == len(pred)

        y_test = joined[getxy.target_column]
        y_test_pred = joined[prediction_variable]
        metrics = {
            # returns (metric_name, metric_value, higher_is_better)
            'test_feval_amex': feval_amex(y_true=y_test, y_pred=y_test_pred)[1],
            'test_feval_amex_gini': feval_amex_gini(y_true=y_test, y_pred=y_test_pred)[1],
            'test_feval_amex_top4': feval_amex_top4(y_true=y_test, y_pred=y_test_pred)[1],
        }
        mlflow.log_metrics(metrics=metrics)

    return metrics


def build_train_objective(
    getxy: GetXY,
    train_data_labelled_batches: List[DataFrame],
    test_data_batches: List[DataFrame],
    id_variables: List[str],
    prediction_variable: str,
    metric_name: str = 'test_feval_amex',
    higher_is_better: bool = True,
):
    def obj_fn(lgb_params):
        metrics = get_cv_hp_metrics(
            getxy=getxy,
            train_data_labelled_batches=train_data_labelled_batches,
            test_data_batches=test_data_batches,
            lgb_params=lgb_params,
            id_variables=id_variables,
            prediction_variable=prediction_variable,
        )
        return {
            'status': hyperopt.STATUS_OK,
            'loss': metrics[metric_name] * -1 if higher_is_better else 1.,
        }
    return obj_fn
