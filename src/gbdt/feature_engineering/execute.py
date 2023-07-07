import pandas as pd
import numpy as np
import polars as pl
from collections import defaultdict
from tqdm.auto import tqdm
from typing import *

from src.data_process import TrainIterator
from src.utils import reduce_memory, Logger
from src.gbdt.feature_engineering import feature_engineering


def execute_feature_engineering(iter_train: TrainIterator, logger: Logger = None) -> Dict[int, pl.DataFrame]:
    all_df = defaultdict(list)
    data_store = defaultdict(pl.DataFrame)

    for session_df in tqdm(iter_train):
        # session_df = session_df.sort(['index']) # sort → frame_sort
        session_id = session_df['session_id'][0]
        level_group = session_df['level_group'][0]
        session_df = reduce_memory(session_df)

        if level_group < 2:
            data_store[session_id] = pl.concat([data_store[session_id], session_df])
            session_df = data_store[session_id]
        else:
            session_df = pl.concat([data_store[session_id], session_df])
            del data_store[session_id]

        features = feature_engineering(session_df, level_group=level_group, use_bingo=True)
        all_df[level_group].append(features)
        
    for i in range(3):
        lg_df = pl.from_pandas(pd.DataFrame(all_df[i]))
        col_num = lg_df.shape[1]
        
        ### null_rate > 0.999を削除
        null_th = 0.999
        cols = lg_df.columns
        null_ratio = lg_df.null_count() / lg_df.shape[0]
        use_cols = [col for col in cols if null_ratio[col][0] <= null_th]
        lg_df = lg_df[use_cols]
        reduce_num = col_num - lg_df.shape[1]
        
        content = f'Level Group={i}, {lg_df.shape}, Feature num after reduction: {reduce_num}'
        print(content)
        if logger is not None:
            logger.log(name='Data Size', content=content)
        
        all_df[i] = lg_df.fill_null(0).fill_nan(0)
        
    return all_df