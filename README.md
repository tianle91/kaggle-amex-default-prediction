# kaggle-amex-default-prediction
https://www.kaggle.com/competitions/amex-default-prediction/


# Useful
`mlflow ui` runs the mlflow ui at [127.0.0.1:5000](http://127.0.0.1:5000)

`kaggle competitions submit -c amex-default-prediction -f ./mlruns/3/afcfa017b3ce4b349e159577f712357c/artifacts/submission.csv -m "afcfa017b3ce4b349e159577f712357c"`


# Submissions
These are taken from [kaggle submissions](https://www.kaggle.com/competitions/amex-default-prediction/submissions).
| index | notebook                     | score | train score | valid score | mlflow run | comments |
| ----- | ---------------------------- | ----- | ----------- | ----------- | ---------- | -------- |
| 9b    | v2_aggregated                | 0.786 | 0.825       | 0.786       | [link](http://127.0.0.1:5000/#/experiments/7/runs/a4afb732fa744b81b8e3163f9af307b0) | same as below, except with raw score |
| 9*    | v2_aggregated                | 0.578 | 0.825       | 0.786       | [link](http://127.0.0.1:5000/#/experiments/7/runs/a4afb732fa744b81b8e3163f9af307b0) | new aggregated features, new batched prediction method |
| 8*    | v2_aggregated                | 0.783 | -           | 0.782       | [link](http://127.0.0.1:5000/#/experiments/7/runs/0261ff0b99024adc88cd4b8566e2a4d1) | - |
| 7*    | use_latest_tune_label_weight | 0.780 | 0.8         | 0.783       | [link](http://127.0.0.1:5000/#/experiments/5/runs/aa3e8418ec5140da995a36e4830290c5) | `negative_label_weight=2.009`achieved best valid score |
| 6*    | use_latest+holiday           | 0.781 | 0.798       | 0.78        | [link](http://127.0.0.1:5000/#/experiments/4/runs/8cc58c1faa8a41ed8ec683f1a0fda6c9) | `postive_label_multiplicative_factor = 0.05` |
| 5     | use_latest+fourier           | 0.776 | 0.794       | 0.775       | [link](http://127.0.0.1:5000/#/experiments/3/runs/afcfa017b3ce4b349e159577f712357c) | `num_boost_round=200` |
| 4     | use_latest                   | 0.773 | 0.779       | 0.777       | [link](http://127.0.0.1:5000/#/experiments/1/runs/74f0f2084c1243788e52c3655f141a35) | `is_unbalance=True` |
| 3*    | use_aggregated               | 0.778 | 0.79        | 0.775       | [link](http://127.0.0.1:5000/#/experiments/2/runs/1e0a4409d0f64b01a242d38c75df61cd) | - |
| 2     | use_latest                   | 0.776 | 0.786       | 0.774       | [link](http://127.0.0.1:5000/#/experiments/1/runs/65418e5e512a433fa7e669bbbeb18880) | - |
| 1     | sample_submission.csv        | 0.019 | -           | -           | - | default prediction is all 0 |

\* Indicates that there is attached commentary.

## Learnings from 9
Same notebook but new aggregated features, new batched prediction method.
Predicted binary instead of raw scores, which has been fixed in 9b.
A lot of max and mean variables made it to the top of the importance charts, which are new compared
to 8.

## Learnings from 8
The difference from 3 (also use_aggregated) is that here we do not set label weights.
Improvement from 7 mainly comes from more features.

## Learnings from 7
A grid search over `negative_label_weight` from 0.01 to 20 resulted very similar train and valid
scores over all the values in the middle of the range, with train scores ~0.8 and valid scores
~0.78.
This is similar to results from the previous iteration, where the equivalent `negative_label_weight`
was 1.

## Learnings from 6
Primarily the improvement didn't come from new features.
The difference between [the previous run](http://127.0.0.1:5000/#/experiments/4/runs/00a14359371847ae9c724d840d06111e)
and [the submitted run](http://127.0.0.1:5000/#/experiments/4/runs/8cc58c1faa8a41ed8ec683f1a0fda6c9)
is `postive_label_multiplicative_factor`.
The previous run set this to 1 whereas the submitted run is 0.05.
Setting it to 0.05 forces equal weights for both positive and negative labels, which is the primary
factor behind the improvement.

Update: Note that setting `postive_label_multiplicative_factor` to 0.05 causes oversampling of
negative labels by 20 times in pre-subsampled training because it was subsampled at 5%.

## Learnings from 3
There were roughly 3 times the original number of features from `transform_aggregated.py` due to
aggregation but did not yield better results compared to only using the latest values.
Similar trends can be observed from the feature importance in [1e0a4409d0f64b01a242d38c75df61cd](http://127.0.0.1:5000/#/experiments/2/runs/1e0a4409d0f64b01a242d38c75df61cd),
where the last observed versions of the same variables dominate, with a few exceptions.


# Data
To download and unzip all files to `data`:
```
kaggle competitions download -c amex-default-prediction -p data
unzip data/amex-default-prediction.zip -d data/amex-default-prediction
python format_data.py
```
It should end up looking like this:
```
data
|-- amex-default-prediction.zip
|-- amex-default-prediction
|   |-- sample_submission.csv
|   |-- sample_submission
|   |   |-- *.parquet
|   |-- test_data.csv
|   |-- test_data
|   |   |-- *.parquet
|   |-- train_data.csv
|   |-- train_data
|   |   |-- *.parquet
|   |-- train_labels.csv
|   |-- train_labels
|   |   |-- *.parquet
```


# Questions
Note that this is the description of how the targets are calculated [src](https://www.kaggle.com/competitions/amex-default-prediction/data):
> The target binary variable is calculated by observing 18 months performance window after the latest credit card statement, and if the customer does not pay due amount in 120 days after their latest
statement date it is considered a default event.
> The dataset contains aggregated profile features for each customer at each statement date. 

## How much history do test ids have?
About the same as train ids. Test ids have no overlap with train ids.
Both train and test periods are about 13 months and the test dates follow immediately after train dates.
However, **the dates in test not uniformly distributed over the 13 months** (as is the case for train dates) and instead oversamples the October to May period.
See [how_much_history.ipynb](notebooks/how_much_history.ipynb).

## Label sequence for each id?
Time between statements has long tails.
The bottom 10% are less than 17 days while top 5% are 392 days.
Number of statements are 13 other than the bottom 10%, which only have 9 statements.
See [label_sequence.ipynb](notebooks/label_sequence.ipynb).

## Predict time to default instead of binary outcome?
Neural network not doing a great job, need to fill the Nones. 
Predicted scale are degenerate with little separation between defaulted and non-defaulted.
See [exploss](exploss.ipynb).

## Add date related features?
TODO:
- sentiments of news articles between 30 days prior to first statement date and last statement date.

## Would KNNImputer improve performance?
[KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
This takes too long, more than 4 hours for train dataset!
We need something more scalable (KNNImputer was only using a single thread).

## Imputing labels
[v2_impute_labels](v2_impute_labels.ipynb)
Applying the default status to all statements didn't improve results (valid set score 0.77).
Applying non-default to statements prior to latest statement gave much worse scores on validation
dataset compared to `use_latest.ipynb` (valid set score 0.67).
