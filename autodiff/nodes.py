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
    def vars(self):
        """
        used to recursivly get all variable from expression
        
        return:
            generator for all variables
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

    def vars(self):
        yield from []

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

    def vars(self):
        yield self.name

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
        
    def vars(self):
        yield from self.left.vars()
        yield from self.right.vars()

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

    def vars(self):
        yield from self.left.vars()
        yield from self.right.vars()

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

    def vars(self):
        yield from self.left.vars()
        yield from self.right.vars()

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

    def vars(self):
        yield from self.left.vars()
        yield from self.right.vars()

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
    def __init__(self, base, exp):
        self.base = base
        self.exp = exp

    def eval(self, env):
        return math.pow(self.base.eval(env), self.exp.eval(env))

    def diff(self, var):
        b = self.base
        db = self.base.diff(var)
        e = self.exp
        de = self.exp.diff(var)
        if type(e) == Const:
            return Multiply(db, Multiply(e, Pow(b, Const(e.eval({})-1))))
        if type(b) == Const:
            return Multiply(de, Multiply(self, Log(b)))
        return Multiply(Pow(b, e), Plus(Multiply(de, Log(b)), Multiply(e, Divide(db, b))))

    def autodiff(self, var, env):
        b = self.base.eval(env)
        db = self.base.autodiff(var, env)
        e = self.exp.eval(env)
        de = self.exp.autodiff(var, env)
        if de == 0:
            return db * e * math.pow(b, e-1)
        if db == 0:
            return de * math.pow(b, e) * math.log(b)
        return math.pow(b, e) * (de * math.log(b) + e * db / b)

    def simplify(self):
        self.base = self.base.simplify()
        self.exp = self.exp.simplify()
        b = self.base
        e = self.exp
        if type(b) == Const:
            self = Const(math.pow(b.eval({}), e.eval({})))
        if e.eval({}) == 1:
            self = b
        if e.eval({}) == 0:
            self = Const(1)
        return self

    def vars(self):
        yield from self.base.vars()
        yield from self.exp.vars()

    def __str__(self):
        if type(self.base) == Const or type(self.base) == Variable:
            b = f"{self.base.__str__()}"
        else:
            b = f"({self.base.__str__()})"
        return f"{b}^{self.exp.__str__()}"

    def to_latex(self):
        if type(self.base) == Const or type(self.base) == Variable:
            b = f"{self.base.to_latex()}"
        else:
            b = f"({self.base.to_latex()})"
        return f"{b}^"+"{"+self.exp.to_latex()+"}"

class Log(Node):
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
        self.child = self.child.simplify()
        return self

    def vars(self):
        yield from self.child.vars()

    def __str__(self):
        return f"ln({self.child.__str__()})"

    def to_latex(self):
        return r"ln("+self.child.to_latex()+")"

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
        self.child = self.child.simplify()
        return self

    def vars(self):
        yield from self.child.vars()

    def __str__(self):
        return f"exp({self.child.__str__()})"

    def to_latex(self):
        return r"\exp("+self.child.to_latex()+")"

class Sin(Node):
    def __init__(self, value):
        self.child = value

    def eval(self, env):
        return math.sin(self.child.eval(env))

    def diff(self, var):
        c = self.child
        dc = self.child.diff(var)
        return Multiply(dc, Cos(c))

    def autodiff(self, var, env):
        c = self.child.eval(env)
        dc = self.child.autodiff(var, env)
        return dc * math.cos(c)

    def simplify(self):
        self.child = self.child.simplify()
        return self

    def vars(self):
        yield from self.child.vars()

    def __str__(self):
        return f"sin({self.child.__str__()})"

    def to_latex(self):
        return r"sin("+self.child.to_latex()+")"

class Cos(Node):
    def __init__(self, value):
        self.child = value

    def eval(self, env):
        return math.cos(self.child.eval(env))

    def diff(self, var):
        c = self.child
        dc = self.child.diff(var)
        return Multiply(dc, Multiply(Const(-1), Sin(c)))

    def autodiff(self, var, env):
        c = self.child.eval(env)
        dc = self.child.autodiff(var, env)
        return -dc * math.sin(c)

    def simplify(self):
        self.child = self.child.simplify()
        return self

    def vars(self):
        yield from self.child.vars()

    def __str__(self):
        return f"cos({self.child.__str__()})"

    def to_latex(self):
        return r"\cos("+self.child.to_latex()+")"

class Tan(Node):
    def __init__(self, value):
        self.child = value

    def eval(self, env):
        return math.tan(self.child.eval(env))

    def diff(self, var):
        c = self.child
        dc = self.child.diff(var)
        return Divide(dc, Pow(Cos(c), Const(2)))

    def autodiff(self, var, env):
        c = self.child.eval(env)
        dc = self.child.autodiff(var, env)
        return dc / math.cos(c)**2

    def simplify(self):
        self.child = self.child.simplify()
        return self

    def vars(self):
        yield from self.child.vars()

    def __str__(self):
        return f"tan({self.child.__str__()})"

    def to_latex(self):
        return r"\tan("+self.child.to_latex()+")"

def draw(self):
    return "$"+fr"{self.to_latex()}"+"$"