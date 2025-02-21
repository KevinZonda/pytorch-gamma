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
        assert x.shape == self.shape, f"Shape mismatch: {x.shape} != {self.shape}"
        return x

class Assert(Lambda):
    def __init__(self, f):
        super().__init__(lambda x: x if f(x) else ValueError("Assertion failed"))

class PrintShape(Lambda):
    def __init__(self):
        super().__init__(lambda x: print(x.shape))

class Print(Lambda):
    def __init__(self):
        super().__init__(lambda x: print(x))
