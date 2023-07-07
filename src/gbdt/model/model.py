import polars as pl
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool, CatBoost


def lgb_train(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_valid: pl.DataFrame,
    y_valid: pl.Series,
    params: dict
):
    feature_name = X_train.columns
    lgb_train = lgb.Dataset(data=X_train.to_numpy(), label=y_train.to_numpy(), feature_name=feature_name)
    lgb_valid = lgb.Dataset(data=X_valid.to_numpy(), label=y_valid.to_numpy(), feature_name=feature_name)

    model = lgb.train(
        params,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        num_boost_round=10000,
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=50)
        ]
    )
    preds = model.predict(X_valid.to_numpy())
    return model, preds


def xgb_train(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_valid: pl.DataFrame,
    y_valid: pl.Series,
    params: dict
):
    feature_name = X_train.columns
    # feature_weights = [10 if col == 'q' else 1 for col in feature_name]
    xgb_train = xgb.DMatrix(X_train.to_numpy(), y_train.to_numpy(), feature_names=feature_name) # feature_weights=feature_weights
    xgb_valid = xgb.DMatrix(X_valid.to_numpy(), y_valid.to_numpy(), feature_names=feature_name)

    model = xgb.train(
        params,
        dtrain=xgb_train,
        evals=[(xgb_train, 'train'), (xgb_valid, 'valid')],
        num_boost_round=10000,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    preds = model.predict(xgb_valid)
    return model, preds

    
def cat_train(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_valid: pl.DataFrame,
    y_valid: pl.Series,
    params: dict
):  
    feature_name = X_train.columns
    cat_train = Pool(X_train.to_numpy(), y_train.to_numpy(), feature_names=feature_name)
    cat_valid = Pool(X_valid.to_numpy(), y_valid.to_numpy(), feature_names=feature_name)
    
    model = CatBoost(params)
    model.fit(
        cat_train,
        eval_set=[cat_train, cat_valid],
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    preds = model.predict(cat_valid, prediction_type='Probability')[:, 1]
    return model, preds