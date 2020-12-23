from .models import DefaultModel, BaseModel
from .packnet import PacknetModel

def get_model(cfg, id):
    if cfg.data.model_name == 'default':
        model = DefaultModel(cfg, id)
    elif cfg.data.model_name == 'base':
        model = BaseModel(cfg, id)
    elif cfg.data.model_name == 'packnet':
        model = PacknetModel(cfg, id)
    else:
        raise NotImplementedError(cfg.data.model_name)
    return model
