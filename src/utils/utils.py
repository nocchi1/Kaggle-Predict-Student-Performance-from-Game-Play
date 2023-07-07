import polars as pl
import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import xgboost as xgb


def reduce_memory(df: pl.DataFrame):
    df = df.drop(['fullscreen', 'hq', 'music'])
    df = df.with_columns([
        pl.col('index').cast(pl.Int32),
        pl.col('level').cast(pl.Int8),
        pl.col('page').fill_null('-1.0').apply(lambda x: x.split('.')[0]).cast(pl.Int8),
        pl.col('room_coor_x').cast(pl.Float32),
        pl.col('room_coor_y').cast(pl.Float32),
        pl.col('screen_coor_x').cast(pl.Float32),
        pl.col('screen_coor_y').cast(pl.Float32),
    ])
    return df


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # False
    
    
# nnの学習で使用する
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# xgboostのfeature importance可視化
def vis_feat_imp(model: xgb.Booster, max_num_features: int):
    _, ax = plt.subplots(figsize=(12,12))
    xgb.plot_importance(model, max_num_features=max_num_features, importance_type='gain', ax=ax)