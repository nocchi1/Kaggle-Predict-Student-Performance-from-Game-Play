import polars as pl
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
import pickle
import gc
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.gbdt.feature_engineering import batch_text_fqid_null_identifier


def convert_cat2idx(lg_df: pl.DataFrame, cat_cols: List[str], level_group: int, save_path: Path):
    for col in cat_cols:
        cat_unique = lg_df.filter(pl.col(col).is_not_null())[col].unique().to_list()
        cat_map = {val: idx for idx, val in enumerate(cat_unique, start=2)} # None,UNK = 1, PAD = 0
        lg_df = lg_df.with_columns([
            pl.col(col).map_dict(cat_map, default=1).alias(col)
        ])
        pickle.dump(cat_map, open(save_path / f'{col}_map_lg{level_group}.pkl', 'wb'))
    return lg_df


def scale_num_feat(lg_df: pl.DataFrame, num_cols: List[str], level_group: int, save_path: Path):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(lg_df[num_cols].to_numpy()) # (n, feat)
    
    lg_df = lg_df.with_columns(
        [pl.Series(scaled_values[:, i]).alias(col) for i, col in enumerate(num_cols)]
    )    
    pickle.dump(scaler, open(save_path / f'numerical_scaler_lg{level_group}.pkl', 'wb'))
    return lg_df


def train_preprocess(config, df: pl.DataFrame):
    df_pp = []
    for level_group, lg_df in df.groupby('level_group'):
        # 前level_groupのデータを使用するか否か
        if config.use_prev_group and level_group > 0:
            lg_prev = df.filter(pl.col('level_group')==(level_group-1))
            lg_prev = lg_prev.sort(['session_id', 'index']).with_columns([
                pl.col('index').cumcount(reverse=True).over('session_id').alias('counts')
            ])
            lg_prev = (
                lg_prev.filter(pl.col('counts') < config.prev_group_window)
                .drop(['counts'])
                .with_columns(
                    [pl.col('level_group') + 1]
                )
            )
            lg_df = pl.concat([lg_prev, lg_df], how='vertical').sort(['session_id', 'index'])
        else:
            lg_df = lg_df.sort(['session_id', 'index'])

        # 前処理
        lg_df = lg_df.with_columns([
            pl.col('elapsed_time').diff().shift(-1).over(['session_id']).fill_null(0).clip(min_val=0, max_val=10 * 60 * 1000).alias('elapsed_time_diff'),
            (pl.col('event_name') + '-' + pl.col('name')).alias('event_name')
        ])
        lg_df = lg_df.with_columns(elapsed_time_diff = np.log1p(pl.col('elapsed_time_diff')))
        lg_df = batch_text_fqid_null_identifier(lg_df, text_col='text_fqid')

        cat_cols = ['event_name', 'text_fqid', 'room_fqid']
        lg_df = convert_cat2idx(lg_df, cat_cols, level_group, save_path=config.output_path)        

        num_cols = ['elapsed_time_diff']
        lg_df = scale_num_feat(lg_df, num_cols, level_group, save_path=config.output_path)
        df_pp.append(lg_df)

    df = pl.concat(df_pp)
    return df


def load_table_feature(data_path: Path):
    table_df = []
    for level_group in range(3):
        df = pl.read_parquet(data_path / f'table_feature_lg{level_group}_with_fs.parquet')
        use_features = pickle.load(open(data_path / f'use_features_lg{level_group}_with_fs.pkl', 'rb'))
        feat_counter = Counter()
        for use_feat in use_features:
            feat_counter.update(use_feat)
        use_features = [col for col, cnt in feat_counter.most_common(1000) if col != 'q']

        df = df[['session_id', 'level_group'] + use_features].fill_null(0).fill_nan(0)
        table_df.append(df)
    return table_df


def preprocess_table_feature(
    config,
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    level_group: int,
    fold: int,
    scaler_type: str
):
    feat_cols = [col for col in train_df.columns if col not in ['session_id', 'level_group']]
    feat_cols = [col for col in feat_cols if 'prev_pred' not in col] # prev_pred系の特徴量を使用するか否か
    
    # get boarder
    max_p, min_p = 99.5, 0.5
    max_limit, min_limit = {}, {}
    for col in feat_cols:
        if 'prev_pred' not in col:
            max_limit[col] = np.percentile(train_df[col], q=max_p)
            min_limit[col] = np.percentile(train_df[col], q=min_p)
        else:
            max_limit[col] = np.max(train_df[col].to_numpy())
            min_limit[col] = np.min(train_df[col].to_numpy())
        
    # clipping
    train_df = train_df.with_columns(
        [pl.col(col).clip(min_val=min_limit[col], max_val=max_limit[col]).alias(col) for col in feat_cols if 'prev_pred' not in col]
    )
    valid_df = valid_df.with_columns(
        [pl.col(col).clip(min_val=min_limit[col], max_val=max_limit[col]).alias(col) for col in feat_cols if 'prev_pred' not in col]
    )

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()

    train_feats = scaler.fit_transform(train_df[feat_cols].to_numpy())
    valid_feats = scaler.transform(valid_df[feat_cols].to_numpy())
    train_feats = dict(zip(train_df['session_id'].to_numpy(), train_feats))
    valid_feats = dict(zip(valid_df['session_id'].to_numpy(), valid_feats))

    pickle.dump(max_limit, open(config.output_path / f'table_max_limit_lg{level_group}_fold{fold}.pkl', 'wb'))
    pickle.dump(min_limit, open(config.output_path / f'table_min_limit_lg{level_group}_fold{fold}.pkl', 'wb'))
    pickle.dump(scaler, open(config.output_path / f'table_scaler_lg{level_group}_fold{fold}.pkl', 'wb'))
    pickle.dump(feat_cols, open(config.output_path / f'table_feat_order_lg{level_group}_fold{fold}.pkl', 'wb'))
    return train_feats, valid_feats