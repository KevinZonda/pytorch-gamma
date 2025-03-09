import torch

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)

class Prob:
    def __init__(self, val=None, shape=(1,), device=None):
        self.val = torch.zeros(shape, device=device) if val is None else to_tensor(val)
        
    def __lshift__(self, other):
        self.val = other._sample(self.shape)
        return self
    
    def __repr__(self):
        return f"Prob(val={self.val}, shape={self.shape})"
    
    @property
    def shape(self):
        return self.val.shape
    
    def to_tensor(self):
        return self.val
    
    @staticmethod
    def from_tensor(t):
        return Prob(val=t)

class Distribution:
    def sample(self):
        pass

class Normal(Distribution):
    def __init__(self, mean, std, reparametrised=False):
        self.mean = to_tensor(mean)
        self.std = to_tensor(std)
        assert self.mean.shape == self.std.shape
        self.reparametrised = reparametrised
    
    def __repr__(self):
        return f"Normal(mean={self.mean}, std={self.std})"
    
    def _sample(self, shape):
        if shape is None:
            shape = self.mean.shape
        if self.reparametrised:
            eps = torch.randn(shape)
            return self.mean + self.std * eps
        else:
            return torch.normal(self.mean, self.std, shape)
    
    def sample(self, shape=None):
        spl = self._sample(shape)
        prob = Prob(val=spl)
        return prob

    def __call__(self, shape):
        return self.sample(shape)

# x = Prob()
# p = Normal(0, 1)
# print(x << p)