import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

class Swap1(Lambda):
    def __init__(self):
        super().__init__(lambda x: x.swapaxes(1, 2))

class Swap2(Lambda):
    def __init__(self):
        super().__init__(lambda x, y: (y, x))

class AssertShape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        assert x.shape == self.shape, f"Shape mismatch. Expected shape {self.shape}, got {x.shape}."
        return x

class AssertShapeLen(Lambda):
    def __init__(self, shape_len):
        super().__init__(lambda x: x if len(x.shape) == shape_len else ValueError(f"Shape mismatch. Expected shape length {shape_len}, got {len(x.shape)}."))

class Assert(Lambda):
    def __init__(self, f, error_message="Assertion failed"):
        super().__init__(lambda x: x if f(x) else ValueError(error_message))

class PrintShape(Lambda):
    def __init__(self):
        super().__init__(lambda x: print(x.shape))

class Print(Lambda):
    def __init__(self):
        super().__init__(lambda x: print(x))

class Fork(nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, x):
        return self.left(x), self.right(x)

class ForkN(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = nn.ModuleList(modules)

    def forward(self, x):
        return [module(x) for module in self.modules]

class Parallel2(nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, x):
        return self.left(x), self.right(x)

class ParallelN(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = nn.ModuleList(modules)

    def forward(self, x):
        return [module(x) for module in self.modules]

def Seq(blocks):
    return nn.Sequential(*blocks)