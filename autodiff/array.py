from abc import abstractmethod
import numpy as np

class Array():
    def __init__(self, value, dtype = 'float32', input:tuple = None, operation = None, track_grads:bool = True, name:str = None):
        from autodiff.operations import Operation
        self._value:np.ndarray = np.array(value, ndmin=1).astype(dtype)
        self._gradient:np.ndarray = None
        self.dimension:tuple = self._value.shape
        self.dtype = self._value.dtype
        self.input:tuple = input
        self.operation:Operation = operation
        self.track_grads:bool = track_grads
        self.name:str = name

    def get_value(self) -> np.ndarray:
        return self._value

    def set_value(self, value):
        v = np.array(value, ndmin=1)
        if not np.issubdtype(v.dtype, np.number):
            raise ValueError("value has to be numeric")
        if self.dimension != v.shape:
            raise ValueError("value shape must match dimension of Expr")
        self._value = v.astype(self.dtype)

    value:np.ndarray = property(get_value, set_value)

    def get_gradient(self) -> np.ndarray:
        return self._gradient

    def set_gradient(self, gradient):
        g = np.array(gradient, ndmin=1)
        if not np.issubdtype(g.dtype, np.number):
            raise ValueError("value has to be numeric")
        if self.dimension != g.shape:
            raise ValueError("gradient shape must match dimension of Expr")
        self._gradient = g.astype(self.dtype)

    gradient:np.ndarray = property(get_gradient, set_gradient)

    def __getitem__(self, key):
        arr = self.value[key]
        if type(arr) == np.ndarray:
            return Array(arr, dtype=self.dtype, track_grads=False)
        return self.value[key]

    def __setitem__(self, key, item):
        self.value[key] = item

    def __delitem__(self, key):
        del self.value[key]

    def eval(self, **env):
        """
        evaluate at given environment
        
        param:
            env : environment, e.g. x=1, y=2
        """
        def iter(node:Array):
            if type(node) != Array:
                return
            for input in (node.input or []):
                iter(input)
            if node.operation != None:
                node.value = node.operation._eval(node.input)
            elif node.name == None:
                return
            elif node.name in env:
                node.value = env[node.name]
        iter(self)

    @abstractmethod
    def diff(self, var:str):
        """
        symbolic differentiation for given variable
        
        param:
            var : variable-name, e.g. "x"
        
        return:
            differentiation result as top level node
        """
        pass

    @abstractmethod
    def autodiff(self, var:str, **env) -> np.ndarray:
        """
        calculates gradient for var at given envirnment, both evaluates expr and calculates diff using forward mode     

        param:
            env : environment dictionary, e.g. x=1, y=2
            var : variable-name to calculate derivative for, e.g. "x"
        
        return:
            gradiant-value
        """
        pass

    @abstractmethod
    def forward(self, var:str) -> np.ndarray:
        """
        calculates gradient for var using autodiff in forward mode

        -> only works after eval has been calculated or rigth after graph creation
        
        param:
            var : variable-name to calculate derivative for, e.g. "x"
        
        return:
            gradiant w.r.t. given variable
        """
        pass

    def backward(self, gradient:np.ndarray=None):
        """
        calculates gradient using autodiff in backward mode

        -> gradient can be found on leaf nodes using expr.gradient \n
        -> only works after eval has been calculated or rigth after graph creation
        """
        if self.dimension == (1,):
            self.gradient = np.ones(self.dimension)
        else:
            self.gradient = gradient

        def iter(node:Array):
            if type(node) != Array:
                return
            if node.operation == None:
                return
            grads = node.operation._backward(node.gradient, node.input)
            for i in range(0, len(node.input)):
                if type(node.input[i]) != Array:
                    continue
                elif node.input[i].operation == None:
                    if node.input[i].gradient == None:
                        node.input[i].gradient = np.zeros(node.input[i].dimension)
                    node.input[i].gradient += grads[i]
                else:
                    node.input[i].gradient = grads[i]
                iter(node.input[i])
        iter(self)

    def __add__(self, p):
        from autodiff.operations import Expand, Add
        a = self
        b = to_array(p)
        if a.dimension != b.dimension:
            if b.dimension == (1,):
                b = Expand.apply(b, a.dimension)
            elif a.dimension == (1,):
                a = Expand.apply(a, b.dimension)
            else:
                raise ValueError("can not add Expr of different dimension")
        return Add.apply(a,b)
    
    def __radd__(self, p):
        from autodiff.operations import Expand, Add
        a = to_array(p)
        b = self
        if a.dimension != b.dimension:
            if b.dimension == (1,):
                b = Expand.apply(b, a.dimension)
            elif a.dimension == (1,):
                a = Expand.apply(a, b.dimension)
            else:
                raise ValueError("can not add Expr of different dimension")
        return Add.apply(a,b)

    def __sub__(self, p):
        from autodiff.operations import Expand, Sub
        a = self
        b = to_array(p)
        if a.dimension != b.dimension:
            if b.dimension == (1,):
                b = Expand.apply(b, a.dimension)
            elif a.dimension == (1,):
                a = Expand.apply(a, b.dimension)
            else:
                raise ValueError("can not add Expr of different dimension")
        return Sub.apply(a,b)

    def __rsub__(self, p):
        from autodiff.operations import Expand, Sub
        a = to_array(p)
        b = self
        if a.dimension != b.dimension:
            if b.dimension == (1,):
                b = Expand.apply(b, a.dimension)
            elif a.dimension == (1,):
                a = Expand.apply(a, b.dimension)
            else:
                raise ValueError("can not add Expr of different dimension")
        return Sub.apply(a,b)

    def __mul__(self, p):
        from autodiff.operations import Expand, Multiply
        a = self
        b = to_array(p)
        if a.dimension != b.dimension:
            if b.dimension == (1,):
                b = Expand.apply(b, a.dimension)
            elif a.dimension == (1,):
                a = Expand.apply(a, b.dimension)
            else:
                raise ValueError("can not add Expr of different dimension")
        return Multiply.apply(a,b)
    
    def __rmul__(self, p):
        from autodiff.operations import Expand, Multiply
        a = to_array(p)
        b = self
        if a.dimension != b.dimension:
            if b.dimension == (1,):
                b = Expand.apply(b, a.dimension)
            elif a.dimension == (1,):
                a = Expand.apply(a, b.dimension)
            else:
                raise ValueError("can not add Expr of different dimension")
        return Multiply.apply(a,b)

    def __truediv__(self, p):
        from autodiff.operations import Expand, Divide
        a = self
        b = to_array(p)
        if a.dimension != b.dimension:
            if b.dimension == (1,):
                b = Expand.apply(b, a.dimension)
            elif a.dimension == (1,):
                a = Expand.apply(a, b.dimension)
            else:
                raise ValueError("can not add Expr of different dimension")
        return Divide.apply(a,b)
    
    def __rtruediv__(self, p):
        from autodiff.operations import Expand, Divide
        a = to_array(p)
        b = self
        if a.dimension != b.dimension:
            if b.dimension == (1,):
                b = Expand.apply(b, a.dimension)
            elif a.dimension == (1,):
                a = Expand.apply(a, b.dimension)
            else:
                raise ValueError("can not add Expr of different dimension")
        return Divide.apply(a,b)
    
    def __pow__(self, p):
        from autodiff.operations import Expand, Pow
        a = self
        b = to_array(p)
        if a.dimension != b.dimension:
            if b.dimension == (1,):
                b = Expand.apply(b, a.dimension)
            elif a.dimension == (1,):
                a = Expand.apply(a, b.dimension)
            else:
                raise ValueError("can not add Expr of different dimension")
        return Pow.apply(a,b)

    def __str__(self):
        return self.value.__str__()

    def _str(self):
        if self.name != None:
            return self.name
        elif self.operation == None:
            return str(self.value)
        else:
            return self.operation._str(self.input)

    def _latex(self):
        if self.name != None:
            return self.name
        elif self.operation == None:
            return str(self.value)
        else:
            return self.operation._latex(self.input)

    def tree(self):
        return Tree(self)


class Tree():
    def __init__(self, expr:Array):
        self.expr = expr

    def __str__(self):
        return self.expr._str()

    def _repr_latex_(self):
        return self.expr._latex()

def to_array(value):
    if type(value) != Array:
        return Array(value, track_grads=True)
    else:
        return value