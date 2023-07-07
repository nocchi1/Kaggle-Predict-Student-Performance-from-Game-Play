from pathlib import Path
import polars as pl
from typing import List


def remove_continue_session(df: pl.DataFrame) -> pl.DataFrame:
    # 終了後もゲームが繰り返される部分を削除
    df = df.with_columns([
        (pl.col('level_group') != pl.col('level_group').shift(1).over('session_id')).alias('change_level_group')
    ])
    df = df.with_columns([
        pl.col('change_level_group').cumsum().over('session_id').alias('change_cumsum')
    ])
    df = df.filter(pl.col('change_cumsum') <= 3)
    df = df.drop(['change_level_group', 'change_cumsum'])
    return df


def sample_by_run_time(df: pl.DataFrame, sample_rate: float) -> pl.DataFrame:
    # run_typeによって使用するデータ量を変更する
    use_num = int(df['session_id'].n_unique() * 0.20)
    use_session = df['session_id'].unique()[:use_num]
    df = df.filter(pl.col('session_id').is_in(use_session))
    return df


def load_train_df(run_type: str, file_path: Path) -> pl.DataFrame:
    """
    学習データ読み込み
    run_typeは、gbdtのみdevでも良さそう、nnでは全データ使用してcv確認したい
    """    
    df = pl.read_parquet(file_path)
    level_group_map = {'0-4': 0, '5-12': 1, '13-22': 2}
    sample_rate_dict = {'half': 0.50, 'debug': 0.01} # 'all': 1.0
    
    if run_type != 'all':
        sample_rate = sample_rate_dict[run_type]
        df = sample_by_run_time(df, sample_rate)

    df = remove_continue_session(df)
    df = df.with_columns(
        level_group = pl.col('level_group').map_dict(level_group_map),
        load_index = pl.col('index').cumcount().over('session_id')
    )
    return df


def load_labels(file_path: Path):
    # ラベルのロード
    labels = pl.read_csv(file_path)
    labels = labels.with_columns(
        q = pl.col('session_id').apply(lambda x: int(x.split('_')[-1][1:])).cast(pl.Int8),
        session_id = pl.col('session_id').apply(lambda x: int(x.split('_')[0]))
    )
    labels = labels.with_columns(
        level_group = pl.col('q').apply(lambda x: 0 if x in range(1,4) else 1 if x in range(4,14) else 2)
    )
    return labels