import torch

class GaussFunction(torch.nn.Module):
  def __init__(self, mean, std):
    super(GaussFunction, self).__init__()

    self.mean = mean
    self.std = std

  def forward(self, x):
    mean = self.mean
    std = self.std
    
    mean = - mean
    x = x + mean
    x = x ** 2
    x = - x

    std = std ** 2
    std = 2 * std
    std = 1 / std
    x = x * std
    x = torch.exp(x)
    return x