from autodiff.nodes import *
import math

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