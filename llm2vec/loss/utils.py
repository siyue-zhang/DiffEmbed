from .HardNegativeNLLLoss import HardNegativeNLLLoss
from .RepllamaLoss import RepllamaLoss

def load_loss(loss_class, *args, **kwargs):
    if loss_class == "HardNegativeNLLLoss":
        loss_cls = HardNegativeNLLLoss
    elif loss_class == 'RepllamaLoss':
        loss_cls = RepllamaLoss
    else:
        raise ValueError(f"Unknown loss class {loss_class}")
    return loss_cls(*args, **kwargs)
