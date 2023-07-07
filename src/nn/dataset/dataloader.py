import polars as pl
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from typing import *
from torch.utils.data import DataLoader

from src.nn.preprocess import preprocess_table_feature
from src.nn.dataset import Collate
from src.nn.dataset import PSPDataset


def get_dataloader(
    config,
    df: pl.DataFrame,
    labels: pl.DataFrame,
    table_df: Optional[pl.DataFrame] = None,
    table_flag: bool = False
):
    all_dataloader = defaultdict(dict)
    gkf = GroupKFold(n_splits=config.fold)
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(X=df, groups=df['session_id'])):
        if fold == config.use_fold:
            break
        
        train_df = df[train_idx, :]
        valid_df = df[valid_idx, :]
        for level_group in range(3):
            train_lg_df = train_df.filter(pl.col('level_group')==level_group)
            valid_lg_df = valid_df.filter(pl.col('level_group')==level_group)
            
            if table_flag:
                # tableデータ準備
                table_lg_df = table_df[level_group]
                train_table = table_lg_df.filter(pl.col('session_id').is_in(train_lg_df['session_id'].unique()))
                valid_table = table_lg_df.filter(pl.col('session_id').is_in(valid_lg_df['session_id'].unique()))
                train_table, valid_table = preprocess_table_feature(
                    config,
                    train_table,
                    valid_table,
                    level_group,
                    fold,
                    scaler_type='minmax'
                )
            else:
                train_table = None
                valid_table = None
            
            # dataloader作成
            collate_fn = Collate(table_flag=table_flag)
            train_loader = DataLoader(
                PSPDataset(
                    config=config,
                    df=train_lg_df.sort(['session_id', 'index']),
                    labels=labels,
                    level_group=level_group,
                    table_flag=table_flag,
                    table_feat=train_table
                ),
                batch_size=config.train_batch,
                collate_fn=collate_fn,
                shuffle=True,
                pin_memory=False,
            )
            valid_loader = DataLoader(
                PSPDataset(
                    config=config,
                    df=valid_lg_df.sort(['session_id', 'index']),
                    labels=labels,
                    level_group=level_group,
                    table_flag=table_flag,
                    table_feat=valid_table,
                ),
                batch_size=config.valid_batch,
                collate_fn=collate_fn,
                shuffle=False,
                pin_memory=False,
            )
            all_dataloader[level_group][fold] = {'train': train_loader, 'valid': valid_loader}

    return all_dataloader