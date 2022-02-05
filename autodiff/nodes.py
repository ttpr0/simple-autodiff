import math
from decimal import Decimal

class Node():
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

class Const(Node):
    def __init__(self, number):
        self.value = Decimal(number.__str__())

    def eval(self, env):
        return self.value

    def diff(self, var):
        return Const(0)

    def autodiff(self, env, var):
        return 0

    def simplify(self):
        return self

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

    def autodiff(self, env, var):
        if self.name == var:
            return 1
        else:
            return 0

    def simplify(self):
        return self

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

    def autodiff(self, env, var):
        return self.left.autodiff(env, var) + self.right.autodiff(env, var)

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        return simplify_Plus(self)

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

    def autodiff(self, env, var):
        return self.left.autodiff(env, var) - self.right.autodiff(env, var)

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        return simplify_minus(self)

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

    def autodiff(self, env, var):
        l = self.left.eval(env)
        dl = self.left.autodiff(env, var)
        r = self.right.eval(env)
        dr = self.right.autodiff(env, var)
        return dl * r + l * dr

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        return simplify_multiply(self)

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

    def autodiff(self, env, var):
        l = self.left.eval(env)
        dl = self.left.autodiff(env, var)
        r = self.right.eval(env)
        dr = self.right.autodiff(env, var)
        return (dl*r - l*dr) / math.pow(r,2)

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        return simplify_divide(self)

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
        return Multiply(dl, Multiply(r, Pow(l, Const(r.eval({})-1))))

    def autodiff(self, env, var):
        l = self.left.eval(env)
        dl = self.left.autodiff(env, var)
        r = self.right.eval(env)
        return dl * r * math.pow(l, r-1)

    def simplify(self):
        self.left = self.left.simplify()
        return simplify_pow(self)

    def __str__(self):
        if type(self.left) == Const or type(self.left) == Variable:
            l = f"{self.left.__str__()}"
        else:
            l = f"({self.left.__str__()})"
        return f"{l}^({self.right.__str__()})"

    def to_latex(self):
        if type(self.left) == Const or type(self.left) == Variable:
            l = f"{self.left.to_latex()}"
        else:
            l = f"({self.left.to_latex()})"
        return f"{l}^"+"{"+self.right.to_latex()+"}"

def draw(node):
    return "$"+fr"{node.to_latex()}"+"$"

def to_node(p):
    if isinstance(p, Node):
        return p
    else:
        return Const(p)

def simplify_Plus(node:Plus):
    l = node.left
    r = node.right
    if type(l) == Const and type(r) == Const:
        node = Const(l.eval({}) + r.eval({}))
    if type(l) == Const and l.eval({}) == 0:
        node = r
    if type(r) == Const and r.eval({}) == 0:
        node = l
    if type(l) == Const:
        if type(r) == Plus:
            if type(r.left) == Const:
                node = Plus(Const(l.eval({})+r.left.eval({})), r.right)
            if type(r.right) == Const:
                node = Plus(Const(l.eval({})+r.right.eval({})), r.left)
        if type(r) == Minus:
            if type(r.left) == Const:
                node = Minus(Const(l.eval({})+r.left.eval({})), r.right)
            if type(r.right) == Const:
                node = Plus(Const(l.eval({})-r.right.eval({})), r.right)
    if type(r) == Const:
        if type(l) == Plus:
            if type(l.left) == Const:
                node = Plus(Const(r.eval({})+l.left.eval({})), l.right)
            if type(l.right) == Const:
                node = Plus(Const(r.eval({})+l.right.eval({})), l.left)
        if type(l) == Minus:
            if type(l.left) == Const:
                node = Minus(Const(r.eval({})+l.left.eval({})), l.right)
            if type(l.right) == Const:
                node = Plus(Const(l.eval({})-l.right.eval({})), r.right)
    return node

def simplify_minus(node:Minus):
    l = node.left
    r = node.right
    if type(l) == Const and type(r) == Const:
        node = Const(l.eval({}) - r.eval({}))
    if type(l) == Const and l.eval({}) == 0:
        node = Const(-r.eval({}))
    if type(r) == Const and r.eval({}) == 0:
        node = l
    if type(l) == Const:
        if type(r) == Plus:
            if type(r.left) == Const:
                node = Minus(Const(l.eval({})-r.left.eval({})), r.right)
            if type(r.right) == Const:
                node = Minus(Const(l.eval({})-r.right.eval({})), r.left)
        if type(r) == Minus:
            if type(r.left) == Const:
                node = Plus(Const(l.eval({})-r.left.eval({})), r.right)
            if type(r.right) == Const:
                node = Minus(Const(l.eval({})+r.right.eval({})), r.right)
    if type(r) == Const:
        if type(l) == Plus:
            if type(l.left) == Const:
                node = Minus(Const(r.eval({})-l.left.eval({})), l.right)
            if type(l.right) == Const:
                node = Minus(Const(r.eval({})-l.right.eval({})), l.left)
        if type(l) == Minus:
            if type(l.left) == Const:
                node = Plus(Const(r.eval({})-l.left.eval({})), l.right)
            if type(l.right) == Const:
                node = Minus(Const(l.eval({})+l.right.eval({})), r.right)
    return node

def simplify_multiply(node:Multiply):
    l = node.left
    r = node.right
    if type(l) == Const and type(r) == Const:
        node = Const(l.eval({}) * l.eval({}))
    if type(l) == Const and l.eval({}) == 0:
        node = Const(0)
    if type(r) == Const and r.eval({}) == 0:
        node = Const(0)
    if type(l) == Const:
        if type(r) == Multiply:
            if type(r.left) == Const:
                node = Multiply(Const(l.eval({})*r.left.eval({})), r.right)
            if type(r.right) == Const:
                node = Multiply(Const(l.eval({})*r.right.eval({})), r.left)
    if type(r) == Const:
        if type(l) == Multiply:
            if type(l.left) == Const:
                node = Multiply(Const(r.eval({})*l.left.eval({})), l.right)
            if type(l.right) == Const:
                node = Multiply(Const(r.eval({})*l.right.eval({})), l.left)
    return node

def simplify_divide(node:Divide):
    l = node.left
    r = node.right
    if type(l) == Const and type(r) == Const:
        node = Const(l.eval({}) / r.eval({}))
    if type(l) == Const and l.eval({}) == 0:
        node = Const(0)
    return node

def simplify_pow(node:Pow):
    l = node.left
    r = node.right
    if type(l) == Const:
        node = Const(math.pow(l.eval({}), r.eval({})))
    if r.eval({}) == 1:
        node = l
    if r.eval({}) == 0:
        node = Const(1)
    return node