import torch.nn as nn
import inspect

def _act(act):
    if isinstance(act, str):
        return _str_to_act(act)
    elif isinstance(act, nn.Module):
        return act
    elif inspect.isclass(act) and issubclass(act, nn.Module):
        return act()
    else:
        raise ValueError(f"Invalid activation function: {act}")

def _str_to_act(act):
    act = act.lower()
    match act:
        case "relu":
            return nn.ReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "gelu":
            return nn.GELU()
        case "elu":
            return nn.ELU()
        case "selu":
            return nn.SELU()
        case "silu":
            return nn.SiLU()
        case "mish":
            return nn.Mish()
        case "softplus":
            return nn.Softplus()
        case "softsign":
            return nn.Softsign()
        case "softmin":
            return nn.Softmin()
        case "softmax":
            return nn.Softmax()
        case _:
            raise ValueError(f"Invalid activation function: {act}")
