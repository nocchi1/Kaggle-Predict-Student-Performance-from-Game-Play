import pandas as pd
import polars as pl
import numpy as np
from typing import *
from tqdm.auto import tqdm
import pickle
from sklearn.model_selection import KFold

from src.gbdt.model import lgb_train, xgb_train, cat_train


def gbdt_train(
    config,
    train_df: pl.DataFrame,
    labels: pl.DataFrame,
    level_group: int,
    use_features_all: Optional[Union[List[List[str]], List[str]]],
) -> Tuple[List[List[Any]], pd.DataFrame, pd.DataFrame]:
    q_range = range(1, 4) if level_group == 0 else range(4, 14) if level_group == 1 else range(14, 19)
    kf = KFold(n_splits=config.fold)

    models = []
    id_lst, q_lst, preds_lst, truths_lst = [], [], [], []
    idx_iter = tqdm(kf.split(X=train_df), total=config.fold)
    for fold, (train_idx, valid_idx) in enumerate(idx_iter):
        train_data = train_df[train_idx, :]
        train_data = pl.concat([train_data.with_columns([pl.lit(q).cast(pl.Int8).alias('q')]) for q in q_range]) # expand along "q"
        train_data = train_data.join(labels[['session_id', 'q', 'correct']], on=['session_id', 'q'], how='left') # join labels

        valid_data = train_df[valid_idx, :]
        valid_data = pl.concat([valid_data.with_columns([pl.lit(q).cast(pl.Int8).alias('q')]) for q in q_range])
        valid_data = valid_data.join(labels[['session_id', 'q', 'correct']], on=['session_id', 'q'], how='left')

        use_features = use_features_all[fold]
        X_train, y_train = train_data[use_features], train_data['correct']
        X_valid, y_valid = valid_data[use_features], valid_data['correct']
        
        if config.model_type == 'lgb':
            model, preds = lgb_train(X_train, y_train, X_valid, y_valid, config.lgb_params)
        elif config.model_type == 'xgb':
            model, preds = xgb_train(X_train, y_train, X_valid, y_valid, config.xgb_params)
        elif config.model_type == 'cat':
            model, preds = cat_train(X_train, y_train, X_valid, y_valid, config.cat_params)
    
        models.append(model)
        id_lst.extend(valid_data['session_id'].to_list())
        q_lst.extend(valid_data['q'].to_list())
        preds_lst.extend(preds.tolist())
        truths_lst.extend(y_valid.to_numpy().tolist())

    oof_df = pl.DataFrame(dict(
        session_id = id_lst,
        q = q_lst,
        pred = preds_lst,
        truth = truths_lst
    ))
    return oof_df, models


def all_level_group_train(
    config,
    all_df: Dict[int, pl.DataFrame],
    all_labels: pl.DataFrame,
    use_feat_dict: Dict[int, List[str]] = None,
    use_prev_pred: bool = False,
    save_model: bool = True,
    train_type: str = None,
):
    models_all, oof_all = [], []
    for level_group in range(3):
        lg_df = all_df[level_group]

        # 過去のlevel_groupの予測 → 特徴量
        if use_prev_pred and (level_group >= 1):
            pred_feat = pl.concat(oof_all)
            pred_feat = pred_feat.pivot(index='session_id', columns='q', values='pred', aggregate_function='first')
            pred_feat.columns = ['session_id'] + [f'prev_pred_q{col}' for col in pred_feat.columns[1:]]
            pred_feat = pred_feat.with_columns(
                prev_pred_mean = pred_feat[:, 1:].mean(axis=1),
                prev_pred_std = pl.Series(pred_feat[:, 1:].to_numpy().std(axis=1)),
                prev_pred_max = pred_feat[:, 1:].max(axis=1),
                prev_pred_min = pred_feat[:, 1:].min(axis=1),
            )
            lg_df = lg_df.join(pred_feat, on=['session_id'], how='left')
        
        # NN用に保存
        lg_df.write_parquet(config.output_path / f'table_feature_lg{level_group}_{train_type}.parquet')

        # 使用する特徴量選択
        if use_feat_dict is not None:
            use_features_all = use_feat_dict[level_group]
        else:
            drop_cols = ['session_id', 'level_group', 'correct']
            use_features = [col for col in lg_df.columns if col not in drop_cols] + ['q']
            use_features_all = [use_features] * config.fold

        oof_df, models = gbdt_train(config, lg_df, all_labels, level_group, use_features_all)
        models_all.append(models)
        oof_all.append(oof_df)
        
        # 使用特徴量とモデル 保存
        if save_model:
            pickle.dump(use_features_all, open(config.output_path / f'use_features_lg{level_group}_{train_type}.pkl', 'wb'))
            pickle.dump(models, open(config.output_path / f'{config.model_type}_models_lg{level_group}_{train_type}.pkl', 'wb'))
        
    oof_df = pl.concat(oof_all).sort(['session_id', 'q'])
    oof_df.write_parquet(config.output_path / f'oof.parquet')
    return oof_df, models_all