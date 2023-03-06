from abc import abstractmethod
import numpy as np

class Array():
    def __init__(self, value, dtype = None, track_grads:bool = False, name:str = None):
        # underlying numpy arrays
        if dtype != None:
            self._value:np.ndarray = np.array(value, ndmin=1).astype(dtype)
        else:
            arr = np.array(value, ndmin=1)
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError("invalid dtype on value")
            self._value = arr
        self._gradient:np.ndarray = None

        # array attributes
        self.name:str = name
        self.track_grads:bool = track_grads

        # computation graph elements
        from autodiff.operations import Operation
        self.operation:Operation = None
        self.input:tuple = None
        self.params:tuple = None

    def get_value(self) -> np.ndarray:
        return self._value

    def set_value(self, value):
        v = np.array(value, ndmin=1)
        if not np.issubdtype(v.dtype, np.number):
            raise ValueError("value has to be numeric")
        if self.shape != v.shape:
            raise ValueError("value shape must match dimension of Expr")
        self._value = v.astype(self.dtype)

    value:np.ndarray = property(get_value, set_value)

    def get_gradient(self) -> np.ndarray:
        return self._gradient

    def set_gradient(self, gradient):
        g = np.array(gradient, ndmin=1)
        if not np.issubdtype(g.dtype, np.number):
            raise ValueError("value has to be numeric")
        if self.shape != g.shape:
            raise ValueError("gradient shape must match dimension of Expr")
        self._gradient = g.astype(self.dtype)

    gradient:np.ndarray = property(get_gradient, set_gradient)

    def get_shape(self) -> tuple:
        return self._value.shape

    shape:tuple = property(get_shape)

    def get_dtype(self) -> tuple:
        return self._value.dtype

    dtype = property(get_dtype)

    def __getitem__(self, key):
        arr = self.value[key]
        if type(arr) == np.ndarray:
            return Array(arr, dtype=self.dtype, track_grads=False)
        return self.value[key]

    def __setitem__(self, key, item):
        if type(item) == Array:
            self.value[key] = item.value
        else:
            self.value[key] = item

    def __delitem__(self, key):
        del self.value[key]

    def eval(self, **env):
        """
        evaluate at given environment
        
        Args:
            env: environment, e.g. x=1, y=2
        """
        def iter(node:Array):
            if type(node) != Array:
                return
            for input in (node.input or []):
                iter(input)
            if node.operation != None:
                input = tuple(item.value if type(item) == Array else item for item in node.input)
                node.value, node.params = node.operation._eval(input)
            elif node.name == None:
                return
            elif node.name in env:
                node.value = env[node.name]
        iter(self)

    @abstractmethod
    def diff(self, var:str):
        """
        symbolic differentiation for given variable
        
        Args:
            var : variable-name, e.g. "x"
        
        Returns:
            differentiation result as top level node
        """
        pass

    @abstractmethod
    def autodiff(self, var:str, **env) -> np.ndarray:
        """
        calculates gradient for var at given envirnment, both evaluates expr and calculates diff using forward mode     

        Args:
            env : environment dictionary, e.g. x=1, y=2
            var : variable-name to calculate derivative for, e.g. "x"
        
        Returns:
            gradiant-value
        """
        pass

    @abstractmethod
    def forward(self, var:str) -> np.ndarray:
        """
        calculates gradient for var using autodiff in forward mode

        -> only works after eval has been calculated or rigth after graph creation
        
        Args:
            var : variable-name to calculate derivative for, e.g. "x"
        
        Returns:
            gradiant w.r.t. given variable
        """
        pass

    def backward(self, gradient:np.ndarray=None):
        """
        calculates gradient using autodiff in backward mode

        -> gradient can be found on leaf nodes using expr.gradient \n
        -> only works after eval has been calculated or rigth after graph creation
        """
        if self.shape == (1,):
            self.gradient = np.ones(self.shape)
        else:
            self.gradient = gradient

        def iter(node:Array):
            if type(node) != Array:
                return
            if node.operation is None:
                return
            input = tuple(item.value if type(item) == Array else item for item in node.input)
            grads = node.operation._backward(node.gradient, input, node.params)
            for i in range(0, len(node.input)):
                if type(node.input[i]) != Array:
                    continue
                if node.input[i].track_grads == False:
                    continue
                elif node.input[i].operation is None:
                    if node.input[i].gradient is None:
                        node.input[i].gradient = np.zeros(node.input[i].shape)
                    node.input[i].gradient += grads[i]
                else:
                    node.input[i].gradient = grads[i]
                iter(node.input[i])
        iter(self)

    def __add__(self, p):
        from autodiff.operations import Add, Expand
        if type(p) == int or type(p) == float:
            p = Array(p)
        if p.shape == (1,):
            return Add.apply(self, Expand.apply(p, self.shape))
        if self.shape == (1,):
            return Add.apply(Expand.apply(self, p.shape), p)
        return Add.apply(self, p)
    
    def __radd__(self, p):
        from autodiff.operations import Add, Expand
        if type(p) == int or type(p) == float:
            p = Array(p)
        if p.shape == (1,):
            return Add.apply(Expand.apply(p, self.shape), self)
        if self.shape == (1,):
            return Add.apply(p, Expand.apply(self, p.shape))
        return Add.apply(p, self)

    def __sub__(self, p):
        from autodiff.operations import Sub, Expand
        if type(p) == int or type(p) == float:
            p = Array(p)
        if p.shape == (1,):
            return Sub.apply(self, Expand.apply(p, self.shape))
        if self.shape == (1,):
            return Sub.apply(Expand.apply(self, p.shape), p)
        return Sub.apply(self, p)

    def __rsub__(self, p):
        from autodiff.operations import Sub, Expand
        if type(p) == int or type(p) == float:
            p = Array(p)
        if p.shape == (1,):
            return Sub.apply(Expand.apply(p, self.shape), self)
        if self.shape == (1,):
            return Sub.apply(p, Expand.apply(self, p.shape))
        return Sub.apply(p, self)

    def __mul__(self, p):
        from autodiff.operations import Multiply, Expand
        if type(p) == int or type(p) == float:
            p = Array(p)
        if p.shape == (1,):
            return Multiply.apply(self, Expand.apply(p, self.shape))
        if self.shape == (1,):
            return Multiply.apply(Expand.apply(self, p.shape), p)
        return Multiply.apply(self, p)
    
    def __rmul__(self, p):
        from autodiff.operations import Multiply, Expand
        if type(p) == int or type(p) == float:
            p = Array(p)
        if p.shape == (1,):
            return Multiply.apply(Expand.apply(p, self.shape), self)
        if self.shape == (1,):
            return Multiply.apply(p, Expand.apply(self, p.shape))
        return Multiply.apply(p, self)

    def __truediv__(self, p):
        from autodiff.operations import Divide, Expand
        if type(p) == int or type(p) == float:
            p = Array(p)
        if p.shape == (1,):
            return Divide.apply(self, Expand.apply(p, self.shape))
        if self.shape == (1,):
            return Divide.apply(Expand.apply(self, p.shape), p)
        return Divide.apply(self, p)
    
    def __rtruediv__(self, p):
        from autodiff.operations import Divide, Expand
        if type(p) == int or type(p) == float:
            p = Array(p)
        if p.shape == (1,):
            return Divide.apply(Expand.apply(p, self.shape), self)
        if self.shape == (1,):
            return Divide.apply(p, Expand.apply(self, p.shape))
        return Divide.apply(p, self)
    
    def __pow__(self, p):
        from autodiff.operations import Pow, Expand
        if type(p) == int or type(p) == float:
            p = Array(p)
        if p.shape == (1,):
            return Pow.apply(self, Expand.apply(p, self.shape))
        if self.shape == (1,):
            return Pow.apply(Expand.apply(self, p.shape), p)
        return Pow.apply(self, p)

    def __matmul__(self, p):
        from autodiff.operations import Matmul
        return Matmul.apply(self,p)

    def __rmatmul__(self, p):
        from autodiff.operations import Matmul
        return Matmul.apply(p,self)

    def T(self):
        from autodiff.operations import Transpose
        return Transpose.apply(self)

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

def from_numpy(arr:np.ndarray, track_grads=False) -> Array:
    return Array(arr, dtype=arr.dtype, track_grads=track_grads)

class Tree():
    def __init__(self, expr:Array):
        self.expr = expr

    def __str__(self):
        return self.expr._str()

    def _repr_latex_(self):
        return self.expr._latex()