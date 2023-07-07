from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def get_optimizer(config, model):
    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    return optimizer
    
def get_scheduler(config, optimizer, init_step):
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=init_step, # 1epochで1周期くらいにする
        T_mult=config.T_mult,
        eta_min=config.eta_min
    )
    return scheduler