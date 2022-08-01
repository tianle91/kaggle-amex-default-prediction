import hyperopt
from evaluation import feval_amex, feval_amex_gini, feval_amex_top4
from sklearn import model_selection
import pandas as pd
from lightgbm import LGBMClassifier
import numpy as np
import mlflow
from typing import List
from pprint import pformat

RANDOM_STATE = 20220731


def get_cv_hp_metrics(
    X_train: pd.DataFrame,
    y_train: np.array,
    X_test: pd.DataFrame,
    y_test: np.array,
    categorical_feature: List[str],
    lgb_params: dict,
) -> dict:
    with mlflow.start_run(nested=True) as run:
        mlflow.log_params(params=lgb_params)
        model = LGBMClassifier(**lgb_params)
        y_train_pred = model_selection.cross_val_predict(
            estimator=model,
            X=X_train,
            y=y_train,
            fit_params={'categorical_feature': categorical_feature},
            method='predict_proba'
        )[:, 1]

        mlflow.lightgbm.autolog()
        model = LGBMClassifier(**lgb_params)
        model.fit(
            X=X_test,
            y=y_test,
            categorical_feature=categorical_feature
        )
        y_test_pred = model.predict_proba(X=X_test)[:, 1]

        metrics = {
            # returns (metric_name, metric_value, higher_is_better)
            'train_feval_amex': feval_amex(y_true=y_train, y_pred=y_train_pred)[1],
            'train_feval_amex_gini': feval_amex_gini(y_true=y_train, y_pred=y_train_pred)[1],
            'train_feval_amex_top4': feval_amex_top4(y_true=y_train, y_pred=y_train_pred)[1],
            'test_feval_amex': feval_amex(y_true=y_test, y_pred=y_test_pred)[1],
            'test_feval_amex_gini': feval_amex_gini(y_true=y_test, y_pred=y_test_pred)[1],
            'test_feval_amex_top4': feval_amex_top4(y_true=y_test, y_pred=y_test_pred)[1],
        }
        mlflow.log_metrics(metrics=metrics)

    return metrics


def build_train_objective(
    X_train: pd.DataFrame,
    y_train: np.array,
    X_test: pd.DataFrame,
    y_test: np.array,
    categorical_feature: List[str],
    metric_name: str = 'test_feval_amex',
    higher_is_better: bool = True,
):
    def obj_fn(lgb_params):
        metrics = get_cv_hp_metrics(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            categorical_feature=categorical_feature,
            lgb_params=lgb_params,
        )
        return {
            'status': hyperopt.STATUS_OK,
            'loss': metrics[metric_name] * -1 if higher_is_better else 1.,
        }
    return obj_fn


def find_best_run(
    run: mlflow.entities.Run,
    metric_name: str = 'test_feval_amex',
    higher_is_better: bool = True,
):
    client = mlflow.tracking.MlflowClient()
    nested_runs = client.search_runs(
        [run.info.experiment_id],
        "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id)
    )
    best_run = min(
        nested_runs,
        key=lambda run: run.data.metrics[metric_name] * -
        1. if higher_is_better else 1.
    )
    mlflow.set_tag("best_run", best_run.info.run_id)
    mlflow.log_metric(f"best_{metric_name}",
                      best_run.data.metrics[metric_name])
    print(
        f'best run id: {best_run.info.run_id} over {len(nested_runs)} runs '
        f'achieved best {metric_name} of {best_run.data.metrics[metric_name]}\n'
        f'params:\n{pformat(best_run.data.params)}\n'
        f'metrics:\n{pformat(best_run.data.metrics)} '
    )
    return best_run
