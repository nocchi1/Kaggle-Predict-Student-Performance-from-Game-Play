import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import *


class PSPDataset(Dataset):
    def __init__(
        self,
        config,
        df: pl.DataFrame,
        labels: pl.DataFrame,
        level_group: int,
        table_flag: bool,
        table_feat: Optional[Dict[int, np.ndarray]],
    ):
        self.config = config
        self.max_len = config.max_len[level_group]
        self.use_cols = ['event_name', 'text_fqid', 'room_fqid', 'elapsed_time_diff']
        self.ids, self.data, self.masks, self.length = self.get_array(df)
        self.labels = labels
        self.level_group = level_group
        self.table_flag = table_flag
        if table_flag:
            self.table_data = table_feat
        

    def __getitem__(self, idx: int):
        session_id = self.ids[idx]
        cat_data = self.data[session_id][:, :3]
        num_data = self.data[session_id][:, 3]
        label_df = self.labels.filter(pl.col('session_id')==session_id).sort(['q'])
        labels = label_df['correct'].to_list()
        q_min = label_df['q'].min()
        questions = (label_df['q'] - q_min).to_list() # 0スタートにする
        mask = self.masks[session_id]
        length = self.length[session_id]

        data_list = [
            session_id,
            torch.tensor(cat_data, dtype=torch.long, device=self.config.device),
            torch.tensor(num_data, dtype=torch.float, device=self.config.device),
            torch.tensor(mask, dtype=torch.float, device=self.config.device),
            torch.tensor(length, dtype=torch.long, device='cpu'),
            torch.tensor(questions, dtype=torch.long, device=self.config.device),
            torch.tensor(labels, dtype=torch.float, device=self.config.device),
        ]
        if self.table_flag:
            data_list.append(
                torch.tensor(self.table_data[session_id], dtype=torch.float, device=self.config.device)
            )
        return data_list

    def __len__(self):
        return len(self.ids)

    def truncate_array(self, session_df: pl.DataFrame):
        feat_array = session_df[self.use_cols].to_numpy()
        # Truncation
        if feat_array.shape[0] > self.max_len: # truncation
            feat_array = feat_array[-self.max_len:, :]

        mask = np.ones(feat_array.shape[0])
        array_len = feat_array.shape[0]
        return feat_array, mask, array_len
    
    def get_array(self, df: pl.DataFrame):
        ids, data, masks, length = [], {}, {}, {}
        for session_id, session_df in df.groupby('session_id'):
            ids.append(session_id)
            feat_array, mask, array_len = self.truncate_array(session_df)
            
            data[session_id] = feat_array
            masks[session_id] = mask
            length[session_id] = array_len
        return ids, data, masks, length