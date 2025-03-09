import torch
import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

class swap1(Lambda):
    def __init__(self):
        super().__init__(lambda x: x.swapaxes(1, 2))

class swap2(Lambda):
    def __init__(self):
        super().__init__(lambda x, y: (y, x))

class assert_shape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        assert x.shape == self.shape, f"Shape mismatch. Expected shape {self.shape}, got {x.shape}."
        return x

class assert_shape_len(Lambda):
    def __init__(self, shape_len):
        super().__init__(lambda x: x if len(x.shape) == shape_len else ValueError(f"Shape mismatch. Expected shape length {shape_len}, got {len(x.shape)}."))

class assert_(Lambda):
    def __init__(self, f, error_message="Assertion failed"):
        super().__init__(lambda x: x if f(x) else ValueError(error_message))

class print_shape(Lambda):
    def __init__(self):
        super().__init__(lambda x: print(x.shape))

class print_(Lambda):
    def __init__(self):
        super().__init__(lambda x: print(x))

class fork(nn.Module):
    def __init__(self, N=2):
        super().__init__()
        self.N = N

    def forward(self, x):
        return [x for _ in range(self.N)]

class parallel(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = nn.ModuleList(modules)

    def forward(self, x):
        return [module(x) for module in self.modules]


def seq(*blocks):
    return nn.Sequential(*blocks)

def _ruby_unbox(block):
    if len(block) == 1:
        return block[0]
    return [ruby_pipeline_element(b) for b in block]

def ruby_pipeline_element(block):
    if isinstance(block, tuple):
        block = seq(*_ruby_unbox(block))
    elif isinstance(block, list):
        block = parallel(*_ruby_unbox(block))
    return block

def pipeline(*blocks):
    return ruby_pipeline_element(blocks)

def Id():
    return nn.Identity()

class apply_at(nn.Module):
    def __init__(self, N, oper):
        super().__init__()
        self.N = N
        self.oper = oper
        
    def forward(self, x):
        x_prime = self.oper(x[self.N])
        x[self.N] = x_prime
        return x

class fst(apply_at):
    def __init__(self, oper):
        super().__init__(0, oper)

class snd(apply_at):
    def __init__(self, oper):
        super().__init__(1, oper)

class lst(nn.Module):
    def __init__(self, N=None):
        super().__init__()
        self.N = N

    def forward(self, x):
        if self.N is None:
            return x[-1]
        return x[-self.N:]

class pick_at(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def forward(self, x):
        return x[self.N]

class drop_at(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def forward(self, x):
        begin = x[:self.N]
        tail = x[self.N + 1:]
        return torch.cat((begin, tail), dim=0)

class fork_inv(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def forward(self, x):
        return x[::-1]

def shape_transform_1d(from_shape, to_shape):
    return nn.Linear(from_shape, to_shape)

def act(act):
    match act:
        case "relu":
            return nn.ReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case _:
            raise ValueError(f"Invalid activation function: {act}")

def activation(act):
    return act(act)

def repeat_block(block, n, same_block=True, block_fn=True, block_args=None, block_kwargs=None):
    if block_fn:
        if not same_block:
            return nn.Sequential(*[block(*block_args, **block_kwargs) for _ in range(n)])
        else:
            block = block(*block_args, **block_kwargs)
    return nn.Sequential(*[block for _ in range(n)])

def relu():
    return nn.ReLU()

def sigmoid():
    return nn.Sigmoid()

def tanh():
    return nn.Tanh()

def softmax():
    return nn.Softmax()

def gelu():
    return nn.GELU()

def silu():
    return nn.SiLU()

def elu():
    return nn.ELU()

def selu():
    return nn.SELU()

def silu():
    return nn.SiLU()

def mish():
    return nn.Mish()
