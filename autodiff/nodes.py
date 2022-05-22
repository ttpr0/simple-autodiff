import math
from decimal import Decimal
from abc import abstractmethod

class Expr():
    def __init__(self, input:tuple, value:float):
        self.dimension:tuple = (1,)
        self.input:tuple = input
        self.value:float = value
        self.gradient:float = 0

    @abstractmethod
    def eval(self, **env) -> float:
        """
        evaluate at given environment
        
        param:
            env : environment, e.g. x=1, y=2
        
        return:
            value
        """
        pass

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
    def autodiff(self, var:str, **env) -> float:
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
    def forward(self, var:str) -> float:
        """
        calculates gradient for var using autodiff in forward mode

        -> only works after eval has been calculated or rigth after graph creation
        
        param:
            var : variable-name to calculate derivative for, e.g. "x"
        
        return:
            gradiant w.r.t. given variable
        """
        pass

    def backward(self):
        """
        calculates gradient using autodiff in backward mode

        -> gradient can be found on leaf nodes using expr.gradient \n
        -> only works after eval has been calculated or rigth after graph creation
        """
        self.gradient = 1

        def iter(node:Expr):
            grads = node.__class__._backward(node.gradient, node.input)
            for i in range(0, len(node.input)):
                if type(node.input[i]) == Variable:
                    node.input[i].gradient += grads[i]
                else:
                    node.input[i].gradient = grads[i]
                iter(node.input[i])
        
        iter(self)

    def __add__(self, p):
        return Add(self,to_node(p))
    
    def __radd__(self, p):
        return Add(to_node(p), self)

    def __sub__(self, p):
        return Sub(self,to_node(p))

    def __rsub__(self, p):
        return Sub(to_node(p), self)

    def __mul__(self, p):
        return Multiply(self,to_node(p))
    
    def __rmul__(self, p):
        return Multiply(to_node(p), self)

    def __truediv__(self, p):
        return Divide(self,to_node(p))
    
    def __rtruediv__(self, p):
        return Divide(to_node(p), self)
    
    def __pow__(self, p):
        return Pow(self,to_node(p))

def to_node(p):
    if isinstance(p, Expr):
        return p
    else:
        return Const(p)

class Const(Expr):
    def __init__(self, number:float, name:str = ""):
        super().__init__((), number)
        self.number:Decimal = Decimal(number)
        self.name:str = name

    def eval(self, **env):
        return self.value

    def diff(self, var):
        return Const(0)

    def autodiff(self, var, **env):
        return 0

    def forward(self, var):
        return 0

    @staticmethod
    def _backward(gradient, input):
        return ()

    def __str__(self):
        return f"{self.number}"

    def _repr_latex_(self):
        return f"{self.number}"

class Variable(Expr):
    def __init__(self, name:str, value:float = 1):
        super().__init__((), value)
        self.name = name

    def eval(self, **env):
        return env[self.name]

    def diff(self, var):
        if self.name == var:
            return Const(1)
        else:
            return Const(0)

    def autodiff(self, var, **env):
        if self.name == var:
            return 1
        else:
            return 0

    def forward(self, var):
        if self.name == var:
            return 1
        else:
            return 0
        
    @staticmethod
    def _backward(gradient, input):
        return ()

    def __str__(self):
        return f"{self.name}"

    def _repr_latex_(self):
        return f"{self.name}"


class Add(Expr):
    def __init__(self, left:Expr, right:Expr):
        super().__init__((left,right), left.value+right.value)

    def eval(self, **env):
        return self.input[0].eval(**env) + self.input[1].eval(**env)

    def diff(self, var):
        return Add(self.input[0].diff(var), self.input[1].diff(var))

    def autodiff(self, var, **env):
        return self.input[0].autodiff(var, **env) + self.input[1].autodiff(var, **env)

    def forward(self, var):
        return self.input[0].forward(var) + self.input[1].forward(var)

    @staticmethod
    def _backward(gradient, input):
        return (gradient,gradient)

    def __str__(self):
        return f"{self.input[0].__str__()} + {self.input[1].__str__()}"
    
    def _repr_latex_(self):
        return f"{self.input[0]._repr_latex_()}+{self.input[1]._repr_latex_()}"

class Sub(Expr):
    def __init__(self, left:Expr, right:Expr):
        super().__init__((left,right), left.value-right.value)

    def eval(self, **env):
        return self.input[0].eval(**env) - self.input[1].eval(**env)

    def diff(self, var):
        return Sub(self.input[0].diff(var), self.input[1].diff(var))

    def autodiff(self, var, **env):
        return self.input[0].autodiff(var, **env) - self.input[1].autodiff(var, **env)

    def forward(self, var):
        return self.input[0].forward(var) - self.input[1].forward(var)

    @staticmethod
    def _backward(gradient, input):
        return (gradient,-gradient)

    def __str__(self):
        return f"{self.input[0].__str__()} - {self.input[1].__str__()}"

    def _repr_latex_(self):
        return f"{self.input[0]._repr_latex_()}-{self.input[1]._repr_latex_()}"

class Multiply(Expr):
    def __init__(self, left:Expr, right:Expr):
        super().__init__((left,right), left.value*right.value)

    def eval(self, **env):
        return self.input[0].eval(**env) * self.input[1].eval(**env)

    def diff(self, var):
        l = self.input[0]
        dl = self.input[0].diff(var)
        r = self.input[1]
        dr = self.input[1].diff(var)
        return Add(Multiply(dl, r), Multiply(l, dr))

    def autodiff(self, var, **env):
        l = self.input[0].eval(**env)
        dl = self.input[0].autodiff(var, **env)
        r = self.input[1].eval(**env)
        dr = self.input[1].autodiff(var, **env)
        return dl * r + l * dr

    def forward(self, var):
        l = self.input[0].value
        dl = self.input[0].forward(var)
        r = self.input[1].value
        dr = self.input[1].forward(var)
        return dl * r + l * dr

    @staticmethod
    def _backward(gradient, input):
        return (gradient*input[1].value,gradient*input[0].value)

    def __str__(self):
        if type(self.input[0]) == Add or type(self.input[0]) == Sub:
            l = f"({self.input[0].__str__()})"
        else:
            l = f"{self.input[0].__str__()}"
        if type(self.input[1]) == Add or type(self.input[1]) == Sub:
            r = f"({self.input[1].__str__()})"
        else:
            r = f"{self.input[1].__str__()}"
        return f"{l} * {r}"        

    def _repr_latex_(self):
        if type(self.input[0]) == Add or type(self.input[0]) == Sub:
            l = f"({self.input[0]._repr_latex_()})"
        else:
            l = f"{self.input[0]._repr_latex_()}"
        if type(self.input[1]) == Add or type(self.input[1]) == Sub:
            r = f"({self.input[1]._repr_latex_()})"
        else:
            r = f"{self.input[1]._repr_latex_()}"
        return f"{l}*{r}" 

class Divide(Expr):
    def __init__(self, left:Expr, right:Expr):
        super().__init__((left,right), left.value/right.value)

    def eval(self, **env):
        return self.input[0].eval(**env) / self.input[1].eval(**env)

    def diff(self, var):
        l = self.input[0]
        dl = self.input[0].diff(var)
        r = self.input[1]
        dr = self.input[1].diff(var)
        return Divide(Sub(Multiply(dl, r), Multiply(l, dr)), Pow(r, Const(2)))

    def autodiff(self, var, **env):
        l = self.input[0].eval(**env)
        dl = self.input[0].autodiff(var, **env)
        r = self.input[1].eval(**env)
        dr = self.input[1].autodiff(var, **env)
        return (dl*r - l*dr) / math.pow(r,2)

    def forward(self, var):
        l = self.input[0].value
        dl = self.input[0].forward(var)
        r = self.input[1].value
        dr = self.input[1].forward(var)
        return (dl*r - l*dr) / math.pow(r,2)

    @staticmethod
    def _backward(gradient, input):
        return (gradient/input[1].value,-gradient*input[0].value/input[1].value**2)

    def __str__(self):
        if type(self.input[0]) == Const or type(self.input[0]) == Variable:
            l = f"{self.input[0].__str__()}"
        else:
            l = f"({self.input[0].__str__()})"
        if type(self.input[1]) == Const or type(self.input[1]) == Variable:
            r = f"{self.input[1].__str__()}"
        else:
            r = f"({self.input[1].__str__()})"
        return f"{l} / {r}"     

    def _repr_latex_(self):
        return r"\frac{"+self.input[0]._repr_latex_()+"}{"+self.input[1]._repr_latex_()+"}"

class Pow(Expr):
    def __init__(self, base:Expr, exp:Expr):
        super().__init__((base,exp), base.value**exp.value)

    def eval(self, **env):
        return math.pow(self.input[0].eval(**env), self.input[1].eval(**env))

    def diff(self, var):
        b = self.input[0]
        db = self.input[0].diff(var)
        e = self.input[1]
        de = self.input[1].diff(var)
        if type(e) == Const:
            return Multiply(db, Multiply(e, Pow(b, Const(e.eval()-1))))
        if type(b) == Const:
            return Multiply(de, Multiply(self, Ln(b)))
        return Multiply(Pow(b, e), Add(Multiply(de, Ln(b)), Multiply(e, Divide(db, b))))

    def autodiff(self, var, **env):
        b = self.input[0].eval(**env)
        db = self.input[0].autodiff(var, **env)
        e = self.input[1].eval(**env)
        de = self.input[1].autodiff(var, **env)
        if de == 0:
            return db * e * math.pow(b, e-1)
        if db == 0:
            return de * math.pow(b, e) * math.log(b)
        return math.pow(b, e) * (de * math.log(b) + e * db / b)

    def forward(self, var):
        b = self.input[0].value
        db = self.input[0].forward(var)
        e = self.input[1].value
        de = self.input[1].forward(var)
        if de == 0:
            return db * e * math.pow(b, e-1)
        if db == 0:
            return de * math.pow(b, e) * math.log(b)
        return math.pow(b, e) * (de * math.log(b) + e * db / b)

    @staticmethod
    def _backward(gradient, input):
        db = input[1].value * input[0].value**(input[1].value-1)
        de = math.log(input[0].value) * input[0].value**input[1].value
        return (gradient*db,gradient*de)

    def __str__(self):
        if type(self.input[0]) == Const or type(self.input[0]) == Variable:
            b = f"{self.input[0].__str__()}"
        else:
            b = f"({self.input[0].__str__()})"
        return f"{b}^{self.input[1].__str__()}"

    def _repr_latex_(self):
        if type(self.input[0]) == Const or type(self.input[0]) == Variable:
            b = f"{self.input[0]._repr_latex_()}"
        else:
            b = f"({self.input[0]._repr_latex_()})"
        return f"{b}^"+"{"+self.input[1]._repr_latex_()+"}"

class Ln(Expr):
    def __init__(self, child:Expr):
        super().__init__((child,), math.log(child.value))

    def eval(self, **env):
        return math.log(self.input[0].eval(**env))

    def diff(self, var):
        c = self.input[0]
        dc = self.input[0].diff(var)
        return Divide(dc, c)

    def autodiff(self, var, **env):
        c = self.input[0].eval(**env)
        dc = self.input[0].autodiff(var, **env)
        return dc / c

    def forward(self, var):
        c = self.input[0].value
        dc = self.input[0].forward(var)
        return dc / c

    @staticmethod
    def _backward(gradient, input):
        return (gradient/input[0].value,)

    def __str__(self):
        return f"ln({self.input[0].__str__()})"

    def _repr_latex_(self):
        return r"ln("+self.input[0]._repr_latex_()+")"