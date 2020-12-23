from . import default_trainer

def get_trainer(cfg):
    if cfg.data.trainer_name == 'default':
        TrainFramework = default_trainer.DefaultTrainer
    else:
        raise NotImplementedError(cfg.data.trainer_name)

    return TrainFramework
