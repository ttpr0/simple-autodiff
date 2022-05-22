from autodiff.nodes import *

class Exp(Expr):
    def __init__(self, child:Expr):
        super().__init__((child,), math.exp(child.value))

    def eval(self, **env):
        return math.exp(self.input[0].eval(**env))

    def diff(self, var):
        c = self.input[0]
        dc = self.input[0].diff(var)
        return Multiply(dc, self)

    def autodiff(self, var, **env):
        c = self.input[0].eval(**env)
        dc = self.input[0].autodiff(var, **env)
        return dc * self.eval(**env)

    def forward(self, var):
        c = self.input[0].value
        dc = self.input[0].forward(var)
        return dc * self.value

    @staticmethod
    def _backward(gradient, input):
        return (gradient*math.exp(input[0].value),)

    def __str__(self):
        return f"exp({self.input[0].__str__()})"

    def _repr_latex_(self):
        return r"\exp("+self.input[0]._repr_latex_()+")"

class Sin(Expr):
    def __init__(self, child:Expr):
        super().__init__((child,), math.sin(child.value))

    def eval(self, **env):
        return math.sin(self.input[0].eval(**env))

    def diff(self, var):
        c = self.input[0]
        dc = self.input[0].diff(var)
        return Multiply(dc, Cos(c))

    def autodiff(self, var, **env):
        c = self.input[0].eval(**env)
        dc = self.input[0].autodiff(var, **env)
        return dc * math.cos(c)

    def forward(self, var):
        c = self.input[0].value
        dc = self.input[0].forward(var)
        return dc * math.cos(c)

    @staticmethod
    def _backward(gradient, input):
        return (gradient*math.cos(input[0].value),)

    def __str__(self):
        return f"sin({self.input[0].__str__()})"

    def _repr_latex_(self):
        return r"sin("+self.input[0]._repr_latex_()+")"

class Cos(Expr):
    def __init__(self, child:Expr):
        super().__init__((child,), math.cos(child.value))

    def eval(self, **env):
        return math.cos(self.input[0].eval(**env))

    def diff(self, var):
        c = self.input[0]
        dc = self.input[0].diff(var)
        return Multiply(dc, Multiply(Const(-1), Sin(c)))

    def autodiff(self, var, **env):
        c = self.input[0].eval(**env)
        dc = self.input[0].autodiff(var, **env)
        return -dc * math.sin(c)

    def forward(self, var):
        c = self.input[0].value
        dc = self.input[0].forward(var)
        return -dc * math.sin(c)

    @staticmethod
    def _backward(gradient, input):
        return (-gradient*math.sin(input[0].value),)

    def __str__(self):
        return f"cos({self.input[0].__str__()})"

    def _repr_latex_(self):
        return r"cos("+self.input[0]._repr_latex_()+")"

class Tan(Expr):
    def __init__(self, child:Expr):
        super().__init__((child,), math.tan(child.value))

    def eval(self, **env):
        return math.tan(self.input[0].eval(**env))

    def diff(self, var):
        c = self.input[0]
        dc = self.input[0].diff(var)
        return Divide(dc, Pow(Cos(c), Const(2)))

    def autodiff(self, var, **env):
        c = self.input[0].eval(**env)
        dc = self.input[0].autodiff(var, **env)
        return dc / math.cos(c)**2

    def forward(self, var):
        c = self.input[0].value
        dc = self.input[0].forward(var)
        return dc / math.cos(c)**2

    @staticmethod
    def _backward(gradient, input):
        return (gradient/math.cos(input[0].value)**2,)

    def __str__(self):
        return f"tan({self.input[0].__str__()})"

    def _repr_latex_(self):
        return r"tan("+self.input[0]._repr_latex_()+")"