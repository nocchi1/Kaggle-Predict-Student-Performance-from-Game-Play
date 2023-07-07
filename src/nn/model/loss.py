import torch
import torch.nn as nn


class AllQuestionLoss(nn.Module):
    def __init__(self, level_group: int, other_rate: float = 0.30):
        super().__init__()
        self.level_group = level_group
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.q_range = range(3) if level_group == 0 else range(3, 13) if level_group == 1 else range(13, 18)
        self.q_range = list(self.q_range)
        self.other_range = [i for i in range(18) if i not in self.q_range]
        self.other_rate = 0.30

    def forward(self, preds, truths):
        preds1 = preds[:, self.q_range]
        truths1 = truths[:, self.q_range]
        loss1 = self.criterion(preds1, truths1)
        
        preds2 = preds[:, self.other_range]
        truths2 = truths[:, self.other_range]
        loss2 = self.criterion(preds2, truths2)
        
        loss = loss1 * (1 - self.other_rate) + loss2 * self.other_rate
        return loss, loss1