import math
from decimal import Decimal
from abc import abstractmethod

class Node():
    @abstractmethod
    def eval(self, env:dict) -> float:
        """
        evaluate at given environment
        
        param:
            env : environment dictionary, e.g. {"x": 1, "y": 2}
        
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
    def autodiff(self, var:str, env:dict) -> float:
        """
        calculates gradient for var at given envirnment
        
        param:
            env : environment dictionary, e.g. {"x": 1, "y": 2}
            var : variable-name to calculate derivative for, e.g. "x"
        
        return:
            gradiant-value
        """
        pass

    @abstractmethod
    def simplify(self):
        """
        simplifies node
        
        return:
            simplified node
        """
        pass

    @abstractmethod
    def get_vars(self, vars:list):
        """
        used to recursivly get all variable from expression
        
        param:
            vars : list of variables, variables found are added to this list
        """
        pass

    def __add__(self, p):
        return Plus(self,to_node(p))
    
    def __radd__(self, p):
        return Plus(to_node(p), self)

    def __sub__(self, p):
        return Minus(self,to_node(p))

    def __rsub__(self, p):
        return Minus(to_node(p), self)

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
    if isinstance(p, Node):
        return p
    else:
        return Const(p)

class Const(Node):
    def __init__(self, number):
        self.value = Decimal(number.__str__())

    def eval(self, env):
        return float(self.value)

    def diff(self, var):
        return Const(0)

    def autodiff(self, var, env):
        return 0

    def simplify(self):
        return self

    def get_vars(self, vars):
        return

    def __str__(self):
        return f"{self.value}"

    def to_latex(self):
        return f"{self.value}"

class Variable(Node):
    def __init__(self, name):
        self.name = name

    def eval(self, env):
        return env[self.name]

    def diff(self, var):
        if self.name == var:
            return Const(1)
        else:
            return Const(0)

    def autodiff(self, var, env):
        if self.name == var:
            return 1
        else:
            return 0

    def simplify(self):
        return self

    def get_vars(self, vars):
        vars.append(self.name)
        return

    def __str__(self):
        return f"{self.name}"

    def to_latex(self):
        return f"{self.name}"

class Plus(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, env):
        return self.left.eval(env) + self.right.eval(env)

    def diff(self, var):
        return Plus(self.left.diff(var), self.right.diff(var))

    def autodiff(self, var, env):
        return self.left.autodiff(var, env) + self.right.autodiff(var, env)

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        l = self.left
        r = self.right
        if type(l) == Const and type(r) == Const:
            self = Const(l.eval({}) + r.eval({}))
        if type(l) == Const and l.eval({}) == 0:
            self = r
        if type(r) == Const and r.eval({}) == 0:
            self = l
        if type(l) == Const:
            if type(r) == Plus:
                if type(r.left) == Const:
                    self = Plus(Const(l.eval({})+r.left.eval({})), r.right)
                if type(r.right) == Const:
                    self = Plus(Const(l.eval({})+r.right.eval({})), r.left)
            if type(r) == Minus:
                if type(r.left) == Const:
                    self = Minus(Const(l.eval({})+r.left.eval({})), r.right)
                if type(r.right) == Const:
                    self = Plus(Const(l.eval({})-r.right.eval({})), r.right)
        if type(r) == Const:
            if type(l) == Plus:
                if type(l.left) == Const:
                    self = Plus(Const(r.eval({})+l.left.eval({})), l.right)
                if type(l.right) == Const:
                    self = Plus(Const(r.eval({})+l.right.eval({})), l.left)
            if type(l) == Minus:
                if type(l.left) == Const:
                    self = Minus(Const(r.eval({})+l.left.eval({})), l.right)
                if type(l.right) == Const:
                    self = Plus(Const(l.eval({})-l.right.eval({})), r.right)
        return self
        
    def get_vars(self, vars):
        self.left.get_vars(vars)
        self.right.get_vars(vars)
        return

    def __str__(self):
        return f"{self.left.__str__()} + {self.right.__str__()}"
    
    def to_latex(self):
        return f"{self.left.to_latex()}+{self.right.to_latex()}"

class Minus(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, env):
        return self.left.eval(env) - self.right.eval(env)

    def diff(self, var):
        return Minus(self.left.diff(var), self.right.diff(var)).simplify()

    def autodiff(self, var, env):
        return self.left.autodiff(var, env) - self.right.autodiff(var, env)

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        l = self.left
        r = self.right
        if type(l) == Const and type(r) == Const:
            self = Const(l.eval({}) - r.eval({}))
        if type(l) == Const and l.eval({}) == 0:
            self = Const(-r.eval({}))
        if type(r) == Const and r.eval({}) == 0:
            self = l
        if type(l) == Const:
            if type(r) == Plus:
                if type(r.left) == Const:
                    self = Minus(Const(l.eval({})-r.left.eval({})), r.right)
                if type(r.right) == Const:
                    self = Minus(Const(l.eval({})-r.right.eval({})), r.left)
            if type(r) == Minus:
                if type(r.left) == Const:
                    self = Plus(Const(l.eval({})-r.left.eval({})), r.right)
                if type(r.right) == Const:
                    self = Minus(Const(l.eval({})+r.right.eval({})), r.right)
        if type(r) == Const:
            if type(l) == Plus:
                if type(l.left) == Const:
                    self = Minus(Const(r.eval({})-l.left.eval({})), l.right)
                if type(l.right) == Const:
                    self = Minus(Const(r.eval({})-l.right.eval({})), l.left)
            if type(l) == Minus:
                if type(l.left) == Const:
                    self = Plus(Const(r.eval({})-l.left.eval({})), l.right)
                if type(l.right) == Const:
                    self = Minus(Const(l.eval({})+l.right.eval({})), r.right)
        return self

    def get_vars(self, vars):
        self.left.get_vars(vars)
        self.right.get_vars(vars)
        return

    def __str__(self):
        return f"{self.left.__str__()} - {self.right.__str__()}"

    def to_latex(self):
        return f"{self.left.to_latex()}-{self.right.to_latex()}"

class Multiply(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, env):
        return self.left.eval(env) * self.right.eval(env)

    def diff(self, var):
        l = self.left
        dl = self.left.diff(var)
        r = self.right
        dr = self.right.diff(var)
        return Plus(Multiply(dl, r), Multiply(l, dr))

    def autodiff(self, var, env):
        l = self.left.eval(env)
        dl = self.left.autodiff(var, env)
        r = self.right.eval(env)
        dr = self.right.autodiff(var, env)
        return dl * r + l * dr

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        l = self.left
        r = self.right
        if type(l) == Const and type(r) == Const:
            self = Const(l.eval({}) * l.eval({}))
        if type(l) == Const and l.eval({}) == 0:
            self = Const(0)
        if type(r) == Const and r.eval({}) == 0:
            self = Const(0)
        if type(l) == Const:
            if type(r) == Multiply:
                if type(r.left) == Const:
                    self = Multiply(Const(l.eval({})*r.left.eval({})), r.right)
                if type(r.right) == Const:
                    self = Multiply(Const(l.eval({})*r.right.eval({})), r.left)
        if type(r) == Const:
            if type(l) == Multiply:
                if type(l.left) == Const:
                    self = Multiply(Const(r.eval({})*l.left.eval({})), l.right)
                if type(l.right) == Const:
                    self = Multiply(Const(r.eval({})*l.right.eval({})), l.left)
        return self

    def get_vars(self, vars):
        self.left.get_vars(vars)
        self.right.get_vars(vars)
        return

    def __str__(self):
        if type(self.left) == Plus or type(self.left) == Minus:
            l = f"({self.left.__str__()})"
        else:
            l = f"{self.left.__str__()}"
        if type(self.right) == Plus or type(self.right) == Minus:
            r = f"({self.right.__str__()})"
        else:
            r = f"{self.right.__str__()}"
        return f"{l} * {r}"        

    def to_latex(self):
        if type(self.left) == Plus or type(self.left) == Minus:
            l = f"({self.left.to_latex()})"
        else:
            l = f"{self.left.to_latex()}"
        if type(self.right) == Plus or type(self.right) == Minus:
            r = f"({self.right.to_latex()})"
        else:
            r = f"{self.right.to_latex()}"
        return f"{l}*{r}" 

class Divide(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, env):
        return self.left.eval(env) / self.right.eval(env)

    def diff(self, var):
        l = self.left
        dl = self.left.diff(var)
        r = self.right
        dr = self.right.diff(var)
        return Divide(Minus(Multiply(dl, r), Multiply(l, dr)), Pow(r, Const(2)))

    def autodiff(self, var, env):
        l = self.left.eval(env)
        dl = self.left.autodiff(var, env)
        r = self.right.eval(env)
        dr = self.right.autodiff(var, env)
        return (dl*r - l*dr) / math.pow(r,2)

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        l = self.left
        r = self.right
        if type(l) == Const and type(r) == Const:
            self = Const(l.eval({}) / r.eval({}))
        if type(l) == Const and l.eval({}) == 0:
            self = Const(0)
        return self

    def get_vars(self, vars):
        self.left.get_vars(vars)
        self.right.get_vars(vars)
        return

    def __str__(self):
        if type(self.left) == Const or type(self.left) == Variable:
            l = f"{self.left.__str__()}"
        else:
            l = f"({self.left.__str__()})"
        if type(self.right) == Const or type(self.right) == Variable:
            r = f"{self.right.__str__()}"
        else:
            r = f"({self.right.__str__()})"
        return f"{l} / {r}"     

    def to_latex(self):
        return r"\frac{"+self.left.to_latex()+"}{"+self.right.to_latex()+"}"

class Pow(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, env):
        return math.pow(self.left.eval(env), self.right.eval(env))

    def diff(self, var):
        l = self.left
        dl = self.left.diff(var)
        r = self.right
        dr = self.right.diff(var)
        if type(r) == Const:
            return Multiply(dl, Multiply(r, Pow(l, Const(r.eval({})-1))))
        if type(l) == Const:
            return Multiply(dr, Multiply(self, Ln(l)))
        return Multiply(Pow(l, r), Plus(Multiply(dr, Ln(l)), Multiply(r, Divide(dl, l))))

    def autodiff(self, var, env):
        l = self.left.eval(env)
        dl = self.left.autodiff(var, env)
        r = self.right.eval(env)
        dr = self.right.autodiff(var, env)
        if dr == 0:
            return dl * r * math.pow(l, r-1)
        if dl == 0:
            return dr * math.pow(l, r) * math.log(l)
        return math.pow(l, r) * (dr * math.log(l) + r * dl / l)

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.right
        l = self.left
        r = self.right
        if type(l) == Const:
            self = Const(math.pow(l.eval({}), r.eval({})))
        if r.eval({}) == 1:
            self = l
        if r.eval({}) == 0:
            self = Const(1)
        return self

    def get_vars(self, vars):
        self.left.get_vars(vars)
        self.right.get_vars(vars)
        return

    def __str__(self):
        if type(self.left) == Const or type(self.left) == Variable:
            l = f"{self.left.__str__()}"
        else:
            l = f"({self.left.__str__()})"
        return f"{l}^{self.right.__str__()}"

    def to_latex(self):
        if type(self.left) == Const or type(self.left) == Variable:
            l = f"{self.left.to_latex()}"
        else:
            l = f"({self.left.to_latex()})"
        return f"{l}^"+"{"+self.right.to_latex()+"}"

class Ln(Node):
    def __init__(self, value):
        self.child = value

    def eval(self, env):
        return math.log(self.child.eval(env))

    def diff(self, var):
        c = self.child
        dc = self.child.diff(var)
        return Divide(dc, c)

    def autodiff(self, var, env):
        c = self.child.eval(env)
        dc = self.child.autodiff(var, env)
        return dc / c

    def simplify(self):
        return self

    def get_vars(self, vars):
        self.child.get_vars(vars)
        return

    def __str__(self):
        return f"ln({self.child.__str__()})"

    def to_latex(self):
        if type(self.left) == Const or type(self.left) == Variable:
            l = f"{self.left.to_latex()}"
        else:
            l = f"({self.left.to_latex()})"
        return f"{l}^"+"{"+self.right.to_latex()+"}"

class Exp(Node):
    def __init__(self, value):
        self.child = value

    def eval(self, env):
        return math.exp(self.child.eval(env))

    def diff(self, var):
        c = self.child
        dc = self.child.diff(var)
        return Multiply(dc, self)

    def autodiff(self, var, env):
        c = self.child.eval(env)
        dc = self.child.autodiff(var, env)
        return dc * self.eval(env)

    def simplify(self):
        return self

    def get_vars(self, vars):
        self.child.get_vars(vars)
        return

    def __str__(self):
        return f"exp({self.child.__str__()})"

    def to_latex(self):
        if type(self.left) == Const or type(self.left) == Variable:
            l = f"{self.left.to_latex()}"
        else:
            l = f"({self.left.to_latex()})"
        return f"{l}^"+"{"+self.right.to_latex()+"}"

def draw(self):
    return "$"+fr"{self.to_latex()}"+"$"