import torch

def _str_to_optim(optm):
    optm = optm.lower()
    match optm:
        case 'adam':
            return torch.optim.Adam
        case 'sgd':
            return torch.optim.SGD
        case 'rmsprop':
            return torch.optim.RMSprop
        case 'adagrad':
            return torch.optim.Adagrad
        case 'adadelta':
            return torch.optim.Adadelta
        case 'adamw':
            return torch.optim.AdamW
        case _:
            raise ValueError(f"Unsupported optimizer name: {optm}")

def _optimizer(model, optimizer, lr=0.01):
    if optimizer is None:
        return torch.optim.Adam(model.parameters(), lr=lr)
    
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
        param_groups = []
        for group in optimizer.param_groups:
            new_group = {k: v for k, v in group.items() if k != 'params'}
            new_group['params'] = model.parameters()
            param_groups.append(new_group)
        return optimizer_type(param_groups)
    
    if isinstance(optimizer, str):
        return _str_to_optim(optimizer)(model.parameters(), lr=lr)
    
    if isinstance(optimizer, dict):
        opt_type = optimizer.get('type', 'adam').lower()
        params = {k: v for k, v in optimizer.items() if k != 'type'}
        return _str_to_optim(opt_type)(model.parameters(), **params)

    else:
        raise ValueError(f"Invalid optimizer type: {type(optimizer)}")
