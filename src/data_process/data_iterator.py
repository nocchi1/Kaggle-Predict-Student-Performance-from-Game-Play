import polars as pl


class TrainIterator:
    # time-series apiを踏襲したiterator
    def __init__(self, df: pl.DataFrame):
        self.df0 = df.filter(pl.col('level_group')==0)
        self.df1 = df.filter(pl.col('level_group')==1)
        self.df2 = df.filter(pl.col('level_group')==2)
                    
    def __iter__(self):
        for _, group_df in self.df0.groupby('session_id'):
            yield group_df
            
        for _, group_df in self.df1.groupby('session_id'):
            yield group_df
            
        for _, group_df in self.df2.groupby('session_id'):
            yield group_df
            
    def __len__(self):
        return (
            self.df0['session_id'].n_unique() + 
            self.df1['session_id'].n_unique() + 
            self.df2['session_id'].n_unique()
        )