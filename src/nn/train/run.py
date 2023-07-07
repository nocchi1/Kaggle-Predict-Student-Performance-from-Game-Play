import pandas as pd
import polars as pl
import pickle
import torch

from src.utils import Logger
from src.metrics import f1_score_macro_optimize_overall, optimize_thresholds, f1_score_macro
from src.nn.model import get_model, get_optimizer, get_scheduler
from src.nn.train import train_model


def run_train_process(config, all_dataloader, org_df: pl.DataFrame, logger: Logger):
    oof_all = []
    for fold in range(config.use_fold):
        oof_df_ = []
        for level_group in range(3):
            train_loader = all_dataloader[level_group][fold]['train']
            valid_loader = all_dataloader[level_group][fold]['valid']

            
            cat_inp_dim = [cat_max + 1 for cat_max in org_df.filter(pl.col('level_group')==level_group)[['event_name', 'text_fqid', 'room_fqid']].max().to_numpy()[0]]
            pickle.dump(cat_inp_dim, open(config.output_path / f'cat_inp_dim_lg{level_group}.pkl', 'wb'))
            if config.table_flag:
                table_dim = len(list(train_loader.dataset.table_data.values())[0])
            else:
                table_dim = None

            model = get_model(config, cat_inp_dim, table_dim, config.table_flag)
            optimizer = get_optimizer(config, model)
            scheduler = get_scheduler(config, optimizer, init_step=len(train_loader))
            
            oof_df, best_loss, best_score, best_th, best_epoch = train_model(config, 
                                                                            model,
                                                                            train_loader,
                                                                            valid_loader,
                                                                            optimizer,
                                                                            scheduler,
                                                                            fold, 
                                                                            level_group,
                                                                            logger,
                                                                            config.table_flag
                                                                            )
            oof_df_.append(oof_df)

            content = f'\t Fold: {fold}, Level Group: {level_group} -->> Best Loss: {best_loss:.4f}, Best Score: {best_score:.4f} ({best_th:.3f}) Best Epoch: {best_epoch} \t'
            if config.log:
                logger.log(name='Evaluation', content=content)
            if config.device == 'cuda':
                torch.cuda.empty_cache()

        oof_df = pd.concat(oof_df_, axis=1)
        preds = oof_df[[f'pred_{i}' for i in range(1, 19)]].to_numpy().reshape(-1)
        truths = oof_df[[f'truth_{i}' for i in range(1, 19)]].to_numpy().reshape(-1)
        best_score, best_th = f1_score_macro_optimize_overall(preds, truths)

        content = '#' * 20 + f'\t Fold: {fold}, -->> OverAll Score: {best_score:.4f} ({best_th:.3f}) \t' + '#' * 20
        if config.log:
                logger.log(name='Fold Evaluation', content=content)
        oof_all.append(oof_df)
    return pd.concat(oof_all)


def create_oof_df(config, oof_df: pd.DataFrame, logger: Logger):
    pred_cols = [col for col in oof_df.columns if 'pred' in col]
    truth_cols = [col for col in oof_df.columns if 'truth' in col]
    oof = oof_df[pred_cols].to_numpy()
    truth = oof_df[truth_cols].to_numpy()
    
    # score確認 & log
    threshold = optimize_thresholds(truth, oof)
    score = f1_score_macro(oof, truth, threshold)
    print(f'best score: {score:.4f}')
    pickle.dump(threshold, open(config.output_path / f'best_th', 'wb'))
    if config.log:
        content = '#' * 20 + f'\t ALL FOLD, -->> Score: {score:.4f}\t' + '#' * 20
        logger.log(name='Overall Evaluation', content=content)

    # # formatを修正してoof保存する
    oof_df = oof_df.reset_index(drop=False).rename({'index': 'session_id'}, axis=1)
    oof_df = pl.from_pandas(oof_df)
    cols = ['session_id'] + oof_df.columns[1:]
    oof_df.columns = cols

    pred_cols = [col for col in oof_df.columns if 'pred' in col]
    preds = oof_df.melt(id_vars='session_id', value_vars=pred_cols)
    preds = (
        preds.with_columns(
            q = pl.col('variable').apply(lambda x: x.split('_')[1]).cast(pl.Int64)
        )
        .rename({'value': 'pred'})
        .drop('variable')
    )

    truth_cols = [col for col in oof_df.columns if 'truth' in col]
    truths = oof_df.melt(id_vars='session_id', value_vars=truth_cols)
    truths = (
        truths.with_columns(
            q = pl.col('variable').apply(lambda x: x.split('_')[1]).cast(pl.Int64)
        )
        .rename({'value': 'truth'})
        .drop('variable')
    )
    oof_df = preds.join(truths, on=['session_id', 'q'], how='left')
    oof_df = oof_df.rename({'pred': 'pred_ryota_nn'})
    oof_df.write_csv(config.output_path / 'oof.csv')