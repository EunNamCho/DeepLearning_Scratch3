import numpy as np
from dezero.core import Function

class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        return gy * cos(self.inputs[0])
    
class Cos(Function):
    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gy):
        return gy * -sin(self.inputs[0])
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        return gy * np.exp(self.inputs[0])
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        return gy * 2 * self.inputs[0]
    
def exp(x):
    return Exp()(x)

def square(x):
    return Square()(x)

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)