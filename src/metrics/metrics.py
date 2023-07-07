import polars as pl
import numpy as np
from typing import *
from sklearn.metrics import f1_score
from scipy.optimize import minimize

from src.utils import Logger


def f1_score_macro(oof: np.ndarray, truth: np.ndarray, thresholds: np.ndarray): # oof.shape = (sample, q)
    oof_binary = (oof > thresholds).astype(int)
    score = f1_score(truth.flatten(), oof_binary.flatten(), average="macro")
    return score


def optimize_thresholds(oof, truth, method='Powell'):
    n_labels = oof.shape[1]
    init_thresholds = np.full(n_labels, 0.60)
    
    objective = lambda thresholds: -1 * f1_score_macro(truth, oof, thresholds)
    result = minimize(objective, init_thresholds, bounds=[(0, 1)] * n_labels, method=method)
    return result.x


def f1_score_macro_optimize_overall(oof: np.ndarray, truth: np.ndarray): # oof.shape = (sample * q)
    best_score = 0.0
    for th in np.arange(0.40, 0.81, 0.01):
        pred = (oof > th).astype('int')
        score = f1_score(truth, pred, average='macro')
        if score > best_score:
            best_score = score
            best_th = th

    return best_score, best_th


def get_score_and_th(oof_df: pl.DataFrame, all_opt_flag: bool, each_opt_flag: bool, logger: Optional[Logger]):
    """
    oof_df : 
        columns: 'session_id', 'q', 'pred', 'truth'
    """
    results = {}
    if all_opt_flag:
        oof = oof_df['pred'].to_numpy()
        truth = oof_df['truth'].to_numpy()
        score, threshold = f1_score_macro_optimize_overall(oof, truth)
        print(f'All Opt -->> best score={score:.4f}, threshold={threshold:.4f}')
        logger.log(name='evaluation', content=f'All Opt, Metric = {score:.4f}, Threshold = {threshold:.3f}')
        results['all_opt_score'] = score
        results['all_opt_th'] = threshold

    if each_opt_flag:
        oof = oof_df.pivot(index='session_id', columns='q', values='pred', aggregate_function='first')[:, 1:].to_numpy()
        truth = oof_df.pivot(index='session_id', columns='q', values='truth', aggregate_function='first')[:, 1:].to_numpy()
        threshold = optimize_thresholds(truth, oof)
        score = f1_score_macro(oof, truth, threshold)
        print(f'Each Opt -->> best score={score:.4f}, threshold={threshold}')
        logger.log(name='evaluation', content=f'Each Opt, Metric = {score:.4f}, Threshold = {threshold}')
        results['each_opt_score'] = score
        results['each_opt_th'] = threshold

    return results