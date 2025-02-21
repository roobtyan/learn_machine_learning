import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

class Convo2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

if __name__ == '__main__':
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    X = X.reshape((1, 1, 6, 8)) # 通道维度、样本维度、高度、宽度
    Y = Y.reshape((1, 1, 6, 8))

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y)**2
        conv2d.zero_grad()
        l.sum().backward()
        conv2d.weight.data -= 3e-2 * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'batch {i + 1}, loss {l.sum():.3f}')
