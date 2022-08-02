import hyperopt
import mlflow
from hyperopt import Trials

from batched import Batched, train_test_split
from format_data import (CATEGORICAL_VARIABLES, DATE_VARIABLES, ID_VARIABLES,
                         PREDICTION_VARIABLE, TARGET_VARIABLE)
from getxy import GetXY
from hp import find_best_run
from hp_batched import build_train_objective
from spark_utils import get_spark_session
from transform_aggregated import (SUMMARY_FEATURE_CATEGORICAL_VARIABLES,
                                  WINDOW_FEATURE_CATEGORICAL_VARIABLES)

spark = get_spark_session()

train_data = spark.read.parquet(
    'data_transformed/amex-default-prediction/train_data_aggregated')
train_labels = spark.read.parquet('data/amex-default-prediction/train_labels')

train_data_labelled = train_data.join(
    train_labels, on=ID_VARIABLES, how='inner')
assert train_data_labelled.count() == train_data.count()
assert train_data_labelled.select(ID_VARIABLES).distinct(
).count() == train_data.select(ID_VARIABLES).distinct().count()


non_feature_columns = [
    TARGET_VARIABLE,
    *ID_VARIABLES,
    *DATE_VARIABLES.keys(),
]
feature_columns = [
    c for c in train_data.columns
    if c not in non_feature_columns
]
print(
    f'Feature columns ({len(feature_columns)}):\n'
    + ', '.join(feature_columns)
)


getxy = GetXY(
    spark=spark,
    feature_columns=feature_columns,
    categorical_columns=[
        *CATEGORICAL_VARIABLES,
        *WINDOW_FEATURE_CATEGORICAL_VARIABLES,
        *SUMMARY_FEATURE_CATEGORICAL_VARIABLES,
    ],
    target_column=TARGET_VARIABLE,
).fit(train_data)


# some rough calculations for batch size
known_good_df = spark.read.parquet(
    'data_transformed/amex-default-prediction/train_data_latest')
known_good_shape = (known_good_df.count(), len(known_good_df.columns))
target_shape = (train_data.count(), len(feature_columns))
batch_size = known_good_df.count() * (len(known_good_df.columns) /
                                      len(feature_columns))
print(f'batch_size: {batch_size}')


fit_data_labelled, test_data_labelled = train_test_split(train_data_labelled)
assert fit_data_labelled.count(
) + test_data_labelled.count() == train_data_labelled.count()


fit_data_labelled_batches = Batched(
    batch_size=batch_size).fit_transform(fit_data_labelled)
test_data_labelled_batches = Batched(
    batch_size=batch_size).fit_transform(test_data_labelled)


MAX_EVALS = 5

space = {
    'class_weight': {
        0.: 1.,
        1.: hyperopt.hp.uniform('class_weight', 0., 10.)
    },
    # 'subsample': hyperopt.hp.uniform('subsample', 0.05, 1.0),
    # The parameters below are cast to int using the scope.int() wrapper
    # 'num_iterations': scope.int(hyperopt.hp.quniform('num_iterations', 10, 200, 1)),
    # 'num_leaves': scope.int(hyperopt.hp.quniform('num_leaves', 20, 50, 1))
}


with mlflow.start_run(nested=False) as run:
    train_objective = build_train_objective(
        getxy=getxy,
        train_data_labelled_batches=fit_data_labelled_batches,
        test_data_batches=test_data_labelled_batches,
        id_variables=ID_VARIABLES,
        prediction_variable=PREDICTION_VARIABLE,
    )
    hyperopt.fmin(
        fn=train_objective,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=MAX_EVALS,
        trials=Trials()
    )
    find_best_run(run)
    print(
        f'run_id: {run.info.run_id} '
        f'experiment_id: {run.info.experiment_id} '
    )

spark.stop()
