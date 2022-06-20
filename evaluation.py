import pandas as pd
from lightgbm import Dataset

from format_data import PREDICTION_VARIABLE, TARGET_VARIABLE


def _amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        # Note that the negative class has been subsampled for this dataset at 5%, and thus receives a 20x weighting in the scoring metric.
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        # Note that the negative class has been subsampled for this dataset at 5%, and thus receives a 20x weighting in the scoring metric.
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d), g, d


def amex_metric(y_true, y_pred) -> float:
    y_true = pd.DataFrame({TARGET_VARIABLE: y_true})
    y_pred = pd.DataFrame({PREDICTION_VARIABLE: y_pred})
    return _amex_metric(y_true=y_true, y_pred=y_pred)


# https://github.com/microsoft/LightGBM/blob/2f5baa3d39efb518cd13a7932fe4d8602c36762f/python-package/lightgbm/engine.py#L54-L71
# return eval_name, eval_result, is_higher_better

def feval_amex(preds, eval_data: Dataset):
    eval_result, _, _ = amex_metric(y_true=eval_data.label, y_pred=preds)
    return 'amex', eval_result, True


def feval_amex_gini(preds, eval_data: Dataset):
    _, eval_result, _ = amex_metric(y_true=eval_data.label, y_pred=preds)
    return 'amex_gini', eval_result, True


def feval_amex_top4(preds, eval_data: Dataset):
    _, _, eval_result = amex_metric(y_true=eval_data.label, y_pred=preds)
    return 'amex_top4', eval_result, True
