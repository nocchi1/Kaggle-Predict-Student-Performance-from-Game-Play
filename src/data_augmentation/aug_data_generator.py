import numpy as np
import polars as pl
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class AugmentedDataGenerator:
    def __init__(self, df: pl.DataFrame, labels: pl.DataFrame):
        self.level_dict = {0: (0, 4), 1: (5, 12), 2: (13, 22)}
        self.df = df.sort(['session_id', 'load_index'])
        self.df = self.df.with_columns(
            elapsed_time_diff = pl.col('elapsed_time').diff().over(['session_id', 'level_group']) # shift(-1)はつけない
        )
        correct = (
            labels.sort(['session_id', 'q'])
            .groupby(['session_id', 'level_group'])
            .agg(
                correct_str = pl.col('correct').cast(pl.Utf8).apply(lambda x: ''.join(list(x)))
            )
        )
        
        # 各level_groupにおいて類似グループの作成
        self.correct_group = {}
        for lg in range(3):
            lg_df = df.filter(pl.col('level_group')==lg)
            lg_df = self._simple_feature_engineering(lg_df)
            lg_df = self._session_clustering(lg_df, lg)
            lg_correct = correct.filter(pl.col('level_group')==lg)
            lg_df = lg_df.join(lg_correct, on=['session_id'], how='left')
            
            group = lg_df.groupby(['cluster_label', 'correct_str']).agg(
                session_id = pl.col('session_id').apply(list),
                session_num = pl.col('session_id').count(),
            )
            group = group.filter(pl.col('session_num') > 1)
            self.correct_group[lg] = group

    def _simple_feature_engineering(self, lg_df: pl.DataFrame):
        lg_df = lg_df.with_columns(
            elapsed_time_diff = pl.col('elapsed_time').diff().shift(-1).over('session_id')
        )

        cat_cols = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid', 'text']
        num_cols = ['elapsed_time_diff','level', 'screen_coor_x', 'screen_coor_y', 'hover_duration']
        
        lg_feat = lg_df.groupby('session_id').agg(
            [pl.col(col).n_unique().alias(f'{col}_nunique') for col in cat_cols] +
            [pl.col(col).mean().alias(f'{col}_mean') for col in num_cols] +
            [pl.col(col).std().alias(f'{col}_std') for col in num_cols] +
            [pl.col(col).max().alias(f'{col}_sum') for col in num_cols] +
            [pl.col(col).max().alias(f'{col}_max') for col in num_cols]
        )
        agg_cat_cols = ['fqid', 'room_fqid', 'text_fqid']
        sum_time_diff = lg_df.pivot(index='session_id', columns=agg_cat_cols, values='elapsed_time_diff', aggregate_function='sum')
        index_count = lg_df.pivot(index='session_id', columns=agg_cat_cols, values='index', aggregate_function='count')
        
        sum_time_diff.columns = [f'{col}_time_sum_{i}' if col != 'session_id' else col for i, col in enumerate(sum_time_diff.columns)]
        index_count.columns = [f'{col}_index_count_{i}' if col != 'session_id' else col for i, col in enumerate(index_count.columns)]

        lg_feat = lg_feat.join(sum_time_diff, on=['session_id'], how='left')
        lg_feat = lg_feat.join(index_count, on=['session_id'])
        return lg_feat.fill_null(0).fill_nan(0)
    
    def _session_clustering(self, lg_df: pl.DataFrame, lg: int, n_clusters: int = 3, sample_th: int = 100):
        scaler = StandardScaler()
        use_cols = [col for col in lg_df.columns if col != 'session_id']
        scaled_value = scaler.fit_transform(lg_df[use_cols].to_numpy())
        # scaled_value[scaled_value == np.inf] = 0
        # scaled_value[scaled_value == -np.inf] = 0

        km = KMeans(n_clusters=n_clusters)
        km.fit(scaled_value)
        cluster_label = km.predict(scaled_value)

        lg_df = lg_df.with_columns(
            cluster_label = pl.Series(cluster_label)
        )
        # level_group2が"text_nunique"で2つに分かれているのを考慮する
        if lg == 2:
            lg_df = lg_df.with_columns(
                text_nunique_group = pl.when(pl.col('text_nunique') <= 190).then(0).otherwise(1)
            )
            lg_df = lg_df.with_columns(
                cluster_label = pl.col('cluster_label') + pl.col('text_nunique_group') * 10
            )

        use_cluster = lg_df['cluster_label'].value_counts().filter(pl.col('counts') > sample_th)['cluster_label'].to_list()
        lg_df = lg_df.filter(pl.col('cluster_label').is_in(use_cluster))
        return lg_df
    
    def generate_data(self, samples: int, level_group: int):
        correct_group = self.correct_group[level_group]
        group_rate = correct_group['session_num'].to_numpy() / correct_group['session_num'].sum()
        session_group = correct_group['session_id'].to_numpy()
        q_range = range(1, 4) if level_group == 0 else range(4, 14) if level_group == 1 else range(14, 19)
        
        new_dfs, new_labels = [], []
        for n in tqdm(range(samples), desc=f'generate augmented data : level_group = {level_group}'):
            group_idx = np.random.choice(range(len(group_rate)), size=1, p=group_rate)[0]
            sessions = correct_group['session_id'][int(group_idx)].to_list()

            idx1, idx2 = np.random.choice(sessions, size=2, replace=False)
            new_df = self._cutmix(idx1, idx2, level_group)
            session_id = str(idx1) + '-' + str(idx2) + '-' + str(n)
            new_df = new_df.with_columns(
                session_id = pl.lit(session_id)
            )

            label = list(map(int, list(correct_group['correct_str'][int(group_idx)])))
            new_label = pl.DataFrame(dict(
                session_id = session_id,
                correct = label,
                q = q_range,
                level_group = level_group
            ))
            new_dfs.append(new_df)
            new_labels.append(new_label)
        
        new_dfs = pl.concat(new_dfs, how='vertical')
        new_labels = pl.concat(new_labels, how='vertical').with_columns(q = pl.col('q').cast(pl.Int8))
        return new_dfs, new_labels
        
    def _cutmix(self, id1: int, id2: int, level_group: int):
        min_level, max_level = self.level_dict[level_group]
        border = np.random.randint(min_level, max_level, size=1)
        
        df1 = (
            self.df
            .filter((pl.col('session_id')==id1) & (pl.col('level_group')==level_group))
            .filter(pl.col('level') <= border)
        )
        df2 = (
            self.df
            .filter((pl.col('session_id')==id2) & (pl.col('level_group')==level_group))
            .filter(pl.col('level') > border)
        )
        df2 = df2.with_columns(
            elapsed_time_cumsum = pl.col('elapsed_time_diff').cumsum()
        )
        df2 = df2.with_columns(
            index = pl.Series(df1['index'][-1] + np.array(range(1, len(df2) + 1))),
            elapsed_time = pl.Series(df1['elapsed_time'][-1] + df2['elapsed_time_cumsum'].to_numpy())
        )
        new_df = pl.concat([df1, df2], how='diagonal')
        assert np.sum(new_df.null_count()[['index', 'elapsed_time']].to_numpy()) == 0
        
        return new_df.drop(['elapsed_time_diff', 'elapsed_time_cumsum'])

    
class AllAugmentedDataGenerator(AugmentedDataGenerator):
    def __init__(self, config, df: pl.DataFrame, labels: pl.DataFrame):
        super().__init__(df, labels)
        self.config = config
        
    def generate_aug_data(self, samples: int, save: bool):
        aug_dfs, aug_labels = [], []
        for level_group in range(3):
            aug_df, aug_label = super().generate_data(samples, level_group)
            aug_dfs.append(aug_df)
            aug_labels.append(aug_label)
            
        aug_df = pl.concat(aug_dfs)
        aug_labels = pl.concat(aug_labels)
        
        if save:
            exist_num = len(list(self.config.input_path.glob('train_augmented_*.parquet')))
            aug_df.write_parquet(self.config.input_path / f'train_augmented_v{exist_num}_samples{samples}.parquet')
            aug_labels.write_parquet(self.config.input_path / f'train_labels_augmented_v{exist_num}_samples{samples}.parquet')

        return aug_df, aug_labels