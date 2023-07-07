import polars as pl
import numpy as np
import gc


def text_fqid_null_identifier(df: pl.DataFrame):
    """
    text_fqidの欠損をどのイベントの後の欠損なのか特定する
    ex.)
        a, None, b, None, c, None
        a, a_NONE, b, b_NONE, c, c_NONE
    """
    text_ids = df['text_fqid'].to_numpy()
    text_ids_shift = np.concatenate([[None], text_ids[:-1]])
    flip_idx = np.where(text_ids != text_ids_shift, True, False)
    null_idx = np.where(text_ids == None, True, False)
    text_ids[flip_idx & null_idx] = [f'{val}_NONE' for val in text_ids_shift[flip_idx & null_idx]]
    
    return df.with_columns(text_fqid = pl.Series(list(text_ids)).fill_null(strategy='forward'))


def batch_text_fqid_null_identifier(df: pl.DataFrame, text_col: str):
    """
    text_fqidの欠損をどのイベントの後の欠損なのか特定する
    ex.)
        a, None, b, None, c, None
        a, a_NONE, b, b_NONE, c, c_NONE
    level_group単位で全てのsessionに対して行う
    """
    df = df.with_columns([
        pl.col(text_col).shift(1).over('session_id').alias(f'{text_col}_diff')
    ])
    df1 = df.filter(~(pl.col(text_col).is_null() & pl.col(f'{text_col}_diff').is_not_null()))
    df2 = df.filter(pl.col(text_col).is_null() & pl.col(f'{text_col}_diff').is_not_null())
    df2 = df2.with_columns([
        (pl.col(f'{text_col}_diff') + '_NONE').alias(text_col)
    ])
    df = pl.concat([df1, df2], how='vertical').sort(['session_id', 'load_index'])
    df = df.with_columns([
        pl.col(text_col).fill_null(strategy='forward').over('session_id')
    ])
    df = df.drop([f'{text_col}_diff'])
    del df1, df2
    gc.collect()
    return df


def create_bingo_feature(features: dict, df: pl.DataFrame, level_group: int):
    # 公式NBで上がっていたbingo-featureを作成する
    if level_group == 1:
        bingo1 = (
            df.filter((pl.col("text") == "Here's the log book.") | (pl.col("fqid") == 'logbook.page.bingo'))
            .select(
                logbook_bingo_duration = pl.col('elapsed_time').max() - pl.col('elapsed_time').min(),
                logbook_bingo_indexCount = pl.col('index').max() - pl.col('index').min()
            )
        )
        bingo2 = (
            df.filter(((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'reader')) | (pl.col("fqid") == "reader.paper2.bingo"))
            .select(
                reader_bingo_duration = pl.col('elapsed_time').max() - pl.col('elapsed_time').min(),
                reader_bingo_indexCount = pl.col('index').max() - pl.col('index').min()
            )
        )
        bingo3 = (
            df.filter(((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'journals')) | (pl.col("fqid") == "journals.pic_2.bingo"))
            .select(
                journals_bingo_duration = pl.col('elapsed_time').max() - pl.col('elapsed_time').min(),
                journals_bingo_indexCount = pl.col('index').max() - pl.col('index').min()
            )
        )
        bingo = pl.concat([bingo1, bingo2, bingo3], how='horizontal')
        bingo = dict(zip(bingo.columns, bingo.to_numpy()[0]))
        features.update(bingo)

    elif level_group == 2:
        bingo1 = (
            df.filter(((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'reader_flag')) | (pl.col("fqid") == "tunic.library.microfiche.reader_flag.paper2.bingo"))
            .select(
                reader_flag_duration = pl.col('elapsed_time').max() - pl.col('elapsed_time').min(),
                reader_flag_indexCount = pl.col('index').max() - pl.col('index').min()
            )
        )
        bingo2 = (
            df.filter(((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'journals_flag')) | (pl.col("fqid") == "journals_flag.pic_0.bingo"))
            .select(
                journalsFlag_bingo_duration = pl.col('elapsed_time').max() - pl.col('elapsed_time').min(),
                journalsFlag_bingo_indexCount = pl.col('index').max() - pl.col('index').min()
            )
        )
        bingo = pl.concat([bingo1, bingo2], how='horizontal')
        bingo = dict(zip(bingo.columns, bingo.to_numpy()[0]))
        features.update(bingo)
        
    return features


def feature_engineering(df: pl.DataFrame, level_group: int, use_bingo: bool = True):
    """
    特徴量エンジニアリングを行う関数
    session単位, level_group単位での処理を行う
    """
    ### preprocess
    df = df.with_columns(
        elapsed_time_diff = pl.col('elapsed_time').diff().shift(-1).clip(0, 1e9).over('session_id'),
        room_coor_move = (pl.col('room_coor_x').diff().shift(-1).over('session_id') ** 2 + pl.col('room_coor_y').diff().shift(-1).over('session_id') ** 2).sqrt().fill_null(0),
        screen_coor_move = (pl.col('screen_coor_x').diff().shift(-1).over('session_id') ** 2 + pl.col('screen_coor_y').diff().shift(-1).over('session_id') ** 2).sqrt().fill_null(0),
        text_fqid_null_flag = pl.col('text_fqid').is_null().cast(pl.Int8)
    )
    df = text_fqid_null_identifier(df)
    df = df.with_columns([
        (pl.col('event_name').fill_null('None') + '_&_' + pl.col('name').fill_null('None')).alias('event_name_&_name'),
        (pl.col('room_fqid').fill_null('None') + '_&_' + pl.col('level').cast(pl.Utf8)).alias('room_fqid_&_level'),
        (pl.col('room_fqid').fill_null('None') + '_&_' + pl.col('fqid').fill_null('None')).alias('room_fqid_&_fqid'),
        (pl.col('room_fqid').fill_null('None') + '_&_' + pl.col('fqid').fill_null('None') + '_&_' + pl.col('level').cast(pl.Utf8)).alias('room_fqid_&_fqid_&_level'),
        (pl.col('room_fqid').fill_null('None') + '_&_' + pl.col('event_name').fill_null('None')).alias('room_fqid_&_event_name'),
        (pl.col('fqid').fill_null('None') + '_&_' + pl.col('event_name').fill_null('None')).alias('fqid_&_event_name'),
        (pl.col('event_name').fill_null('None') + '_&_' + pl.col('room_fqid').fill_null('None') + '_&_' + pl.col('text_fqid').fill_null('None')).alias('event_name_&_room_fqid_&_text_fqid'),
    ])

    ### features作成
    features = {}
    features['session_id'] = df['session_id'][0]
    features['level_group'] = level_group
    
    ### 基本的な特徴量
    features['event_count'] = len(df)
    features['elapsed_time_duration'] = df['elapsed_time'].max() - df['elapsed_time'].min()
    features['text_unique_rate'] = df['text'].n_unique() / len(df)
    features['elapsed_time_per_event'] = features['elapsed_time_duration'] / features['event_count']
    
    ### 基本的な特徴量2
    cat_cols = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid', 'text']
    num_cols = ['elapsed_time_diff', 'level', 'page', 'hover_duration', 'room_coor_x',
                'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'room_coor_move', 'screen_coor_move']

    for col in cat_cols:
        features[f'{col}_nunique'] = df[col].n_unique()
        features[f'{col}_duplicate'] = sum(df[col].shift() == df[col])

    for col in num_cols:
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_sum'] = df[col].sum()
        features[f'{col}_max'] = df[col].max()
        
    ### Aggregation
    agg_cols = ['level', 'event_name', 'room_fqid', 'fqid', 'room_fqid_&_level', 'room_fqid_&_event_name']
    if level_group >= 1:
        agg_cols.append('level_group')

    for col in agg_cols:
        agg_feat = df.groupby(col).agg([
            pl.col('index').count().alias('index_count'),
            pl.col('elapsed_time_diff').mean().alias('elapsed_time_diff_mean'),
            pl.col('elapsed_time_diff').std().alias('elapsed_time_diff_std'),
            pl.col('elapsed_time_diff').sum().alias('elapsed_time_diff_sum'),
            pl.col('room_coor_move').mean().alias('room_coor_move_mean'),
            pl.col('room_coor_move').std().alias('room_coor_move_std'),
            pl.col('room_coor_move').sum().alias('room_coor_move_sum'),
            pl.col('hover_duration').sum().alias('hover_duration_sum'),
        ])
        agg_feat = agg_feat.with_columns(
            elapsed_time_diff_rate = pl.col('elapsed_time_diff_sum') / features['elapsed_time_duration']
        )
        agg_feat = agg_feat.melt(id_vars=[col], value_vars=agg_feat.columns[1:])
        agg_feat = agg_feat.with_columns(
            col_name = f'{col}_' + pl.col(col).cast(pl.Utf8) + '_' + pl.col('variable')
        )
        agg_feat = dict(agg_feat[['col_name', 'value']].to_numpy())
        features.update(agg_feat)

    ### Aggregation2
    agg_cols2 = ['text_fqid', 'text', 'room_fqid_&_fqid', 'event_name_&_room_fqid_&_text_fqid']
    for col in agg_cols2:
        agg_feat = df.groupby(col).agg([
            pl.col('index').count().alias('index_count'),
            pl.col('elapsed_time_diff').sum().alias('elapsed_time_diff_sum'),
            pl.col('room_coor_move').sum().alias('room_coor_move_sum'),
            pl.col('hover_duration').sum().alias('hover_duration_sum'),
            pl.col('text_fqid_null_flag').first().alias('text_fqid_null_flag')
            
        ])
        if col in ['event_name_&_room_fqid_&_text_fqid']:
            agg_feat = agg_feat.filter(pl.col('text_fqid_null_flag')==1)
            
        agg_feat = agg_feat.drop('text_fqid_null_flag')
        agg_feat = agg_feat.with_columns(
            elapsed_time_diff_rate = pl.col('elapsed_time_diff_sum') / features['elapsed_time_duration']
        )
        agg_feat = agg_feat.melt(id_vars=[col], value_vars=agg_feat.columns[1:])
        agg_feat = agg_feat.with_columns(
            col_name = f'{col}_' + pl.col(col).cast(pl.Utf8) + '_' + pl.col('variable')
        )
        agg_feat = dict(agg_feat[['col_name', 'value']].to_numpy())
        features.update(agg_feat)

    ### bingo feature
    if use_bingo and (level_group >= 1):
        features = create_bingo_feature(features, df, level_group)
        
    return features