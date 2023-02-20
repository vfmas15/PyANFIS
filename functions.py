import torch

def gauss(mean, std, x):
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