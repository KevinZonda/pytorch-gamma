import torch

def to_tensor(x, device=None, requires_grad=False):
    if isinstance(x, torch.Tensor):
        if device is not None and x.device != device:
            x = x.to(device)
        return x
    return torch.tensor(x, device=device, requires_grad=requires_grad)

class Prob(torch.Tensor):
    @staticmethod
    def __new__(cls, val=None, shape=(1,), device=None, requires_grad=False):
        if val is None:
            val = torch.zeros(shape, device=device, requires_grad=requires_grad)
        else:
            val = to_tensor(val, device=device, requires_grad=requires_grad)
        return torch.Tensor._make_subclass(cls, val, device_for_backend_keys=device, require_grad=requires_grad)
        
    def __lshift__(self, other):
        new_val = other._sample(self.shape)
        self.copy_(new_val)
        return self
    
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

x = Prob()
p = Normal(0, 1)
print(x << p)