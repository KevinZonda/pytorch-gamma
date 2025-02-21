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

class fork_n(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def forward(self, x):
        return [x for _ in range(self.N)]

class fork(fork_n):
    def __init__(self):
        super().__init__(2)

class parallel_n(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = nn.ModuleList(modules)

    def forward(self, x):
        return [module(x) for module in self.modules]

class parallel2(parallel_n):
    def __init__(self, left, right):
        super().__init__(left, right)


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
        block = parallel_n(*_ruby_unbox(block))
    return block

def ruby_pipeline(*blocks):
    return ruby_pipeline_element(blocks)

def Id():
    return nn.Identity()

class apply_n(nn.Module):
    def __init__(self, N, oper):
        super().__init__()
        self.N = N
        self.oper = oper
        
    def forward(self, x):
        x_prime = self.oper(x[self.N])
        x[self.N] = x_prime
        return x

class fst(apply_n):
    def __init__(self, oper):
        super().__init__(0, oper)

class snd(apply_n):
    def __init__(self, oper):
        super().__init__(1, oper)

class fork_inv(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def forward(self, x):
        return x[::-1]

def shape_transform_1d(from_shape, to_shape):
    return nn.Linear(from_shape, to_shape)

def activation(act):
    match act:
        case "relu":
            return nn.ReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case _:
            raise ValueError(f"Invalid activation function: {act}")
