import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch

from src.utils import AverageMeter
from src.metrics import f1_score_macro_optimize_overall
from src.nn.model import AllQuestionLoss


def train_model(config, model, train_loader, valid_loader, optimizer, scheduler, fold, level_group, logger, table_flag):
    best_loss = np.inf
    loss_fn = AllQuestionLoss(level_group=level_group)
    q_range = range(1, 4) if level_group == 0 else range(4, 14) if level_group == 1 else range(14, 19)
    
    for epoch in tqdm(range(config.epochs), desc=f'Fold: {fold}, Level Group: {level_group} -->> Training'):
        ### train
        model.train()
        meter = AverageMeter()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            if table_flag:
                ids, cat_feat, num_feat, mask, length, questions, labels, table_feat = data
                outputs = model(cat_feat, num_feat, table_feat, mask, questions, length)
            else:
                ids, cat_feat, num_feat, mask, length, questions, labels = data
                outputs = model(cat_feat, num_feat, mask, questions, length)

            loss, lg_loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            meter.update(lg_loss.item(), config.train_batch)
            
        # log train messeage
        lr = scheduler.get_last_lr()[0]
        content = f'Train[Level Group: {level_group}, Fold: {fold}, Epoch: {epoch + 1}] -->>  Loss: {meter.avg:.4f}, LR: {lr:.8f}'
        print(content)
        if config.log:
            logger.log(name='Train Model', content=content)

        ### validation
        model.eval()
        meter = AverageMeter()
        
        session_ids = []
        preds = []
        truths = []
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                if table_flag:
                    ids, cat_feat, num_feat, mask, length, questions, labels, table_feat = data
                    outputs = model(cat_feat, num_feat, table_feat, mask, questions, length)
                else:
                    ids, cat_feat, num_feat, mask, length, questions, labels = data
                    outputs = model(cat_feat, num_feat, mask, questions, length)

                loss, lg_loss = loss_fn(outputs, labels)
                meter.update(lg_loss.item(), config.valid_batch)
                outputs = 1 / (1 + torch.exp(-1 * outputs)) # (batch, 3)
                session_ids.extend(list(ids))
                preds.append(outputs.detach().cpu().numpy())
                truths.append(labels.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)[:, np.array(q_range) - 1]
        truths = np.concatenate(truths, axis=0)[:, np.array(q_range) - 1]
        
        score, th = f1_score_macro_optimize_overall(preds.reshape(-1,), truths.reshape(-1,)) # reshape(-1,) → ndarray
        loss_epoch = meter.avg
        
        # log valid messeage
        content = f'Valid[Level Group: {level_group}, Fold: {fold}, Epoch: {epoch + 1}] -->>  Loss: {meter.avg:.4f}, Score: {score:.4f}, Threshold: {th:.4f}'
        print(content)
        if config.log:
            logger.log(name='Train Model', content=content)

        if best_loss > loss_epoch:
            torch.save(model.state_dict(), config.output_path / f'model_fold{fold}_lg{level_group}.pth')
            best_loss = loss_epoch
            best_score = score
            best_th = th
            best_epoch = epoch

            oof_df = pd.concat([
                # idの保存
                pd.DataFrame(preds, columns=[f'pred_{q}' for q in q_range]),
                pd.DataFrame(truths, columns=[f'truth_{q}' for q in q_range], dtype=np.int8),
            ], axis=1)
            oof_df.index = session_ids

    return oof_df, best_loss, best_score, best_th, best_epoch