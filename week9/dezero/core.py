import numpy as np
import weakref
import contextlib
from typing import List, Tuple


class Config:
    enable_backprop = True

def as_array(data):
    if np.isscalar(data):
        return np.array(data)
    return data

def as_varialbe(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config("enable_backprop", False)

class Variable:
    def __init__(self, data:np.array, name=None):
        if not data.any():
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}은(는) 지원하지 않습니다.")
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            xs, ys = f.inputs, f.outputs
            gys = [y().grad for y in ys]

            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = gxs,

                for x, gx in zip(xs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x. grad = x.grad + gx
                
                    if x.creator is not None:
                        add_func(x.creator)
                
                if not retain_grad:
                    for y in ys:
                        y().grad = None

    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def shape(self):
        return self.data.shape
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ")"


class Function:
    def __call__(self, *inputs:List[Variable]):
        inputs = [as_varialbe(as_array(input)) for input in inputs]

        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = ys,
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creator(self)
            self.outputs = [weakref.ref(output) for output in outputs]
            self.inputs = inputs
            
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        return NotImplementedError()
    
    def backward(self, gy):
        return NotImplementedError()
    
class Add(Function):
    def forward(self, x0, x1):
        #x1 = as_array(x1)
        return x0 + x1,

    def backward(self, gy):
        return gy, gy
    
class Mul(Function):
    def forward(self, x0, x1):
        #x1 = as_array(x1)
        return x0 * x1
    
    def backward(self, gy):
        return gy * self.inputs[1], gy * self.inputs[0]
    
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1
    
    def backward(self, gy):
        return gy, -gy
    
class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1
    
    def backward(self, gy):
        return gy / self.inputs[1], -gy * self.inputs[0] / (self.inputs[1] ** 2)
    
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        return gy * self.c * self.inputs[0] ** (self.c - 1)

def add(x0, x1):
    return Add()(x0, x1)

def mul(x0, x1):
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    return Sub()(x0, x1)

def div(x0, x1):
    return Div()(x0, x1)

def pow(x, c):
    return Pow(c)(x)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = sub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = div
    Variable.__pow__ = pow
