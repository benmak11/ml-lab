# Intro to ML, lesson 4: Model Validation

Kaggle Learn, [Model Validation](https://www.kaggle.com/code/dansbecker/model-validation).
Code: `notebooks/00-warmup/tutorial_2.py`.

## The idea

A model's value is its predictions on data it hasn't seen, so measure it there.
Scoring a model on the same rows it was trained on (an "in-sample" score) mostly
measures how well it memorised them.

**Mean absolute error (MAE):** the average of |actual − predicted| over all rows.
It reads in the target's units: "on average, predictions are off by about $X."

**Why in-sample scores lie:** the model learns any pattern in the training data,
including ones that are chance. Kaggle's example is that green-door houses
happened to be expensive in the sample. Those patterns look accurate in training
and fail on new data.

**The fix:** hold some rows out. `train_test_split(X, y, random_state=...)` returns
`train_X, val_X, train_y, val_y` (75% / 25% by default). Fit on the training rows
only, predict on `val_X`, and score against `val_y`.

## Melbourne results (`tutorial_2.py`)

| Score | MAE |
| --- | --- |
| In-sample | $435 |
| Validation | $262,494 |

The average validation price is about $1.09M, so the real error is about a
quarter of a house's price. The near-zero in-sample number was memorisation.
Kaggle's version gets $265,807 because its tree has no `random_state`.

## Exercise (Iowa housing, done on Kaggle)

The Iowa data (`home-data-for-ml-course/train.csv`) is only on Kaggle, so this
runs there, not locally. Features: `LotArea`, `YearBuilt`, `1stFlrSF`,
`2ndFlrSF`, `FullBath`, `BedroomAbvGr`, `TotRmsAbvGrd`; target `SalePrice`.

1. Split: `train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)`
2. Fit on training rows only: `DecisionTreeRegressor(random_state=1).fit(train_X, train_y)`
3. Predict on validation rows: `val_predictions = iowa_model.predict(val_X)`
4. Score: `val_mae = mean_absolute_error(val_y, val_predictions)`. Validation MAE: _fill in_

The in-sample predictions matched the first five prices exactly; the validation
predictions don't. That gap is the lesson.

## Gotchas I hit

- `mean_absolute_error` takes the true values first, then the predictions.
- A NumPy array has no `.head()`; slice it with `[:5]`. A pandas Series has `.head()`.
- Fit on `train_X, train_y`, not `X, y`, or the split does nothing.

## Next

Underfitting and Overfitting: use validation MAE to pick a tree size (`max_leaf_nodes`).
