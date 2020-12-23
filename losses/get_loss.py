from .losses import DefaultLoss, BaseLoss

def get_loss(cfg, id):
    if cfg.data.loss_name == 'default':
        loss = DefaultLoss(cfg, id)
    elif cfg.data.loss_name == "base":
        loss = BaseLoss(cfg, id)
    else:
        raise NotImplementedError(cfg.data.loss_name)
    return loss
