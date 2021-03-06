from autodiff.array import Array
from abc import abstractmethod
import numpy as np

class Operation():

    @classmethod
    def apply(cls, *input):
        cls._validate_input(input)
        value = cls._eval(input)
        if any(i.track_grads for i in input):
            input = tuple(input)
            return Array(input=input, value=value, operation=cls)
        else:
            return Array(value, track_grads=False)

    @staticmethod
    @abstractmethod
    def _validate_input(input):
        pass

    @staticmethod
    @abstractmethod
    def _eval(input):
        pass

    @staticmethod
    @abstractmethod
    def _diff(input, gradient):
        pass

    @staticmethod
    @abstractmethod
    def _forward(input):
        pass

    @staticmethod
    @abstractmethod
    def _backward(gradient, input):
        pass

    @staticmethod
    @abstractmethod
    def _str(input):
        pass

    @staticmethod
    @abstractmethod
    def _latex(input):
        pass


class Add(Operation):
    @staticmethod
    def _validate_input(input):
        if input[0].dimension != input[1].dimension:
            raise ValueError("dimensions do not match")

    @staticmethod
    def _eval(input):
        return input[0].value + input[1].value

    @staticmethod
    def _diff(input, gradient):
        # return Add(self.input[0].diff(var), self.input[1].diff(var))
        pass

    @staticmethod
    def _forward(input):
        # return self.input[0].forward(var) + self.input[1].forward(var)
        pass

    @staticmethod
    def _backward(gradient, input):
        return (gradient,gradient)

    @staticmethod
    def _str(input):
        return f"{input[0]._str()} + {input[1]._str()}"
    
    @staticmethod
    def _latex(input):
        return f"{input[0]._latex()}+{input[1]._latex()}"


class Sub(Operation):
    @staticmethod
    def _validate_input(input):
        if input[0].dimension != input[1].dimension:
            raise ValueError("dimensions do not match")

    @staticmethod
    def _eval(input):
        return input[0].value - input[1].value

    @staticmethod
    def _diff(input, gradient):
        # return Sub(self.input[0].diff(var), self.input[1].diff(var))
        pass

    @staticmethod
    def _forward(input):
        # return self.input[0].forward(var) - self.input[1].forward(var)
        pass

    @staticmethod
    def _backward(gradient, input):
        return (gradient,-gradient)

    @staticmethod
    def _str(input):
        return f"{input[0]._str()} - {input[1]._str()}"

    @staticmethod
    def _latex(input):
        return f"{input[0]._latex()}-{input[1]._latex()}"


class Multiply(Operation):
    @staticmethod
    def _validate_input(input):
        if input[0].dimension != input[1].dimension:
            raise ValueError("dimensions do not match")

    @staticmethod
    def _eval(input):
        return input[0].value * input[1].value

    @staticmethod
    def _diff(input, gradient):
        # l = self.input[0]
        # dl = self.input[0].diff(var)
        # r = self.input[1]
        # dr = self.input[1].diff(var)
        # return Add(Multiply(dl, r), Multiply(l, dr))
        pass

    @staticmethod
    def _forward(input):
        # l = self.input[0].value
        # dl = self.input[0].forward(var)
        # r = self.input[1].value
        # dr = self.input[1].forward(var)
        # return dl * r + l * dr
        pass

    @staticmethod
    def _backward(gradient, input):
        return (gradient*input[1].value,gradient*input[0].value)

    @staticmethod
    def _str(input):
        if input[0].operation == Add or input[0].operation == Sub:
            l = f"({input[0]._str()})"
        else:
            l = f"{input[0]._str()}"
        if input[1].operation == Add or input[1].operation == Sub:
            r = f"({input[1]._str()})"
        else:
            r = f"{input[1]._str()}"
        return f"{l} * {r}"        

    @staticmethod
    def _latex(input):
        if input[0].operation == Add or input[0].operation == Sub:
            l = f"({input[0]._latex()})"
        else:
            l = f"{input[0]._latex()}"
        if input[1].operation == Add or input[1].operation == Sub:
            r = f"({input[1]._latex()})"
        else:
            r = f"{input[1]._latex()}"
        return f"{l}*{r}" 

class Divide(Operation):
    @staticmethod
    def _validate_input(input):
        if input[0].dimension != input[1].dimension:
            raise ValueError("dimensions do not match")

    @staticmethod
    def _eval(input):
        return input[0].value / input[1].value

    @staticmethod
    def _diff(input, gradient):
        # l = self.input[0]
        # dl = self.input[0].diff(var)
        # r = self.input[1]
        # dr = self.input[1].diff(var)
        # return Divide(Sub(Multiply(dl, r), Multiply(l, dr)), Pow(r, Param(2)))
        pass

    @staticmethod
    def _forward(input):
        # l = self.input[0].value
        # dl = self.input[0].forward(var)
        # r = self.input[1].value
        # dr = self.input[1].forward(var)
        # return (dl*r - l*dr) / r**2
        pass

    @staticmethod
    def _backward(gradient, input):
        return (gradient/input[1].value,-gradient*input[0].value/input[1].value**2)

    @staticmethod
    def _str(input):
        if input[0].operation == None:
            l = f"{input[0]._str()}"
        else:
            l = f"({input[0]._str()})"
        if input[1].operation == None:
            r = f"{input[1]._str()}"
        else:
            r = f"({input[1]._str()})"
        return f"{l} / {r}"     

    @staticmethod
    def _latex(input):
        return r"\frac{"+input[0]._latex()+"}{"+input[1]._latex()+"}"

class Pow(Operation):
    @staticmethod
    def _validate_input(input):
        if input[0].dimension != input[1].dimension:
            raise ValueError("dimensions do not match")

    @staticmethod
    def _eval(input):
        return input[0].value ** input[1].value

    @staticmethod
    def _diff(input, gradient):
        # b = self.input[0]
        # db = self.input[0].diff(var)
        # e = self.input[1]
        # de = self.input[1].diff(var)
        # if type(e) == Param:
        #     return Multiply(db, Multiply(e, Pow(b, Param(e.eval()-1))))
        # if type(b) == Param:
        #     return Multiply(de, Multiply(self, Ln(b)))
        # return Multiply(Pow(b, e), Add(Multiply(de, Ln(b)), Multiply(e, Divide(db, b))))
        pass

    @staticmethod
    def _forward(input):
        # b = self.input[0].value
        # db = self.input[0].forward(var)
        # e = self.input[1].value
        # de = self.input[1].forward(var)
        # if de == 0:
        #     return db * e * b**(e-1)
        # if db == 0:
        #     return de * b**e * np.log(b)
        # return b**e * (de * np.log(b) + e * db / b)
        pass

    @staticmethod
    def _backward(gradient, input):
        db = input[1].value * input[0].value**(input[1].value-1)
        de = np.log(input[0].value) * input[0].value**input[1].value
        return (gradient*db,gradient*de)

    @staticmethod
    def _str(input):
        if input[0].operation == None:
            b = f"{input[0]._str()}"
        else:
            b = f"({input[0]._str()})"
        return f"{b}^({input[1]._str()})"

    @staticmethod
    def _latex(input):
        if input[0].operation == None:
            b = f"{input[0]._latex()}"
        else:
            b = f"({input[0]._latex()})"
        return f"{b}^"+"{"+input[1]._latex()+"}"

class Ln(Operation):
    @staticmethod
    def _validate_input(input):
        return

    @staticmethod
    def _eval(input):
        return np.log(input[0].value)

    @staticmethod
    def _diff(input, gradient):
        # c = self.input[0]
        # dc = self.input[0].diff(var)
        # return Divide(dc, c)
        pass

    @staticmethod
    def _forward(input):
        # c = self.input[0].value
        # dc = self.input[0].forward(var)
        # return dc / c
        pass

    @staticmethod
    def _backward(gradient, input):
        return (gradient/input[0].value,)

    @staticmethod
    def _str(input):
        return f"ln({input[0]._str()})"

    @staticmethod
    def _latex(input):
        return r"ln("+input[0]._latex()+")"

class Expand(Operation):
    @staticmethod
    def _validate_input(input):
        if input[0].dimension != (1,):
            raise ValueError("only scalar can be expanded")
        if type(input[1]) != tuple:
            raise ValueError("dimension not valid")

    @staticmethod
    def _eval(input):
        return np.full(input[1], input[0].value)

    @staticmethod
    def _diff(input, gradient):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input):
        return (np.array(np.sum(gradient)),)

    @staticmethod
    def _str(input):
        return f"{input[0]._str()}"
    
    @staticmethod
    def _latex(input):
        return f"{input[0]._latex()}"

def ln(child:Array):
    return Ln.apply(child)

def expand(child:Array, dimension:tuple):
    return Expand.apply(child, dimension)

class Exp(Operation):
    @staticmethod
    def _validate_input(input):
        return

    @staticmethod
    def _eval(input):
        return np.exp(input[0].value)

    @staticmethod
    def _diff(inpit, gradient):
        # c = self.input[0]
        # dc = self.input[0].diff(var)
        # return Multiply(dc, self)
        pass

    @staticmethod
    def _forward(input):
        # c = self.input[0].value
        # dc = self.input[0].forward(var)
        # return dc * self.value
        pass

    @staticmethod
    def _backward(gradient, input):
        return (gradient*np.exp(input[0].value),)

    @staticmethod
    def _str(input):
        return f"exp({input[0]._str()})"

    @staticmethod
    def _latex(input):
        return r"\exp("+input[0]._latex()+")"

class Sin(Operation):
    @staticmethod
    def _validate_input(input):
        return 

    @staticmethod
    def _eval(input):
        return np.sin(input[0].value)

    @staticmethod
    def _diff(input, gradient):
        # c = self.input[0]
        # dc = self.input[0].diff(var)
        # return Multiply(dc, Cos(c))
        pass

    @staticmethod
    def _forward(input):
        # c = self.input[0].value
        # dc = self.input[0].forward(var)
        # return dc * np.cos(c)
        pass

    @staticmethod
    def _backward(gradient, input):
        return (gradient*np.cos(input[0].value),)

    @staticmethod
    def _str(input):
        return f"sin({input[0]._str()})"

    @staticmethod
    def _latex(input):
        return r"sin("+input[0]._latex()+")"

class Cos(Operation):
    @staticmethod
    def _validate_input(input):
        return

    @staticmethod
    def _eval(input):
        return np.cos(input[0].value)

    @staticmethod
    def _diff(input, gradient):
        # c = self.input[0]
        # dc = self.input[0].diff(var)
        # return Multiply(dc, Multiply(Param(-1), Sin(c)))
        pass

    @staticmethod
    def _forward(input):
        # c = self.input[0].value
        # dc = self.input[0].forward(var)
        # return -dc * np.sin(c)
        pass

    @staticmethod
    def _backward(gradient, input):
        return (-gradient*np.sin(input[0].value),)

    @staticmethod
    def _str(input):
        return f"cos({input[0]._str()})"

    @staticmethod
    def _latex(input):
        return r"cos("+input[0]._latex()+")"

class Tan(Operation):
    @staticmethod
    def _validate_input(input):
        return

    @staticmethod
    def _eval(input):
        return np.tan(input[0].value)

    @staticmethod
    def _diff(input, gradient):
        # c = self.input[0]
        # dc = self.input[0].diff(var)
        # return Divide(dc, Pow(Cos(c), Param(2)))
        pass

    @staticmethod
    def _forward(input):
        # c = self.input[0].value
        # dc = self.input[0].forward(var)
        # return dc / np.cos(c)**2
        pass

    @staticmethod
    def _backward(gradient, input):
        return (gradient/np.cos(input[0].value)**2,)

    @staticmethod
    def _str(input):
        return f"tan({input[0]._str()})"

    @staticmethod
    def _latex(input):
        return r"tan("+input[0]._latex()+")"

class MeanSquaredError(Operation):
    @staticmethod
    def _validate_input(input):
        if input[0].dimension != input[1].dimension:
            raise ValueError("target dimension must match output dimesion")
        return

    @staticmethod
    def _eval(input):
        v = (input[0].value - input[1].value)**2
        return np.array([np.sum(v)/v.size])

    @staticmethod
    def _diff(input, gradient):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input):
        v = (input[0].value - input[1].value) * 2
        return (gradient*v/v.size,-gradient*v/v.size)

    @staticmethod
    def _str(input):
        return f"error({input[0]._str()})"

    @staticmethod
    def _latex(input):
        return r"error("+input[0]._latex()+")"

class Transpose(Operation):
    @staticmethod
    def _validate_input(input):
        return

    @staticmethod
    def _eval(input):
        return np.transpose(input[0].value)

    @staticmethod
    def _diff(input):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input):
        return (np.transpose(gradient),)

    @staticmethod
    def _str(input):
        return f"{input[0]._str()}.T"

    @staticmethod
    def _latex(input):
        return r""+input[0]._latex()+"^T"

class Inv(Operation):
    @staticmethod
    def _validate_input(input):
        if len(input[0].dimension) != 2 or input[0].dimension[0] != input[0].dimension[1]:
            raise ValueError("inv only allowed for quadratic matricies")
        return

    @staticmethod
    def _eval(input):
        return np.linalg.inv(input[0].value)

    @staticmethod
    def _diff(input, gradient):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input):
        inv = np.linalg.inv(input[0].value)
        v = -np.matmul(np.transpose(inv), inv)
        return (np.matmul(gradient, v),)

    @staticmethod
    def _str(input):
        return f"inv({input[0]._str()})"

    @staticmethod
    def _latex(input):
        return r""+input[0]._latex()+"^-1"

class Matmul(Operation):
    @staticmethod
    def _validate_input(input):
        if len(input[0].dimension) != 2 or len(input[1].dimension) != 2:
            raise ValueError("matmul only for 2d matricies")
        if input[0].dimension[1] != input[1].dimension[0]:
            raise ValueError("dimesion error")  
        return

    @staticmethod
    def _eval(input):
        return np.matmul(input[0].value, input[1].value)

    @staticmethod
    def _diff(input, gradient):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input):
        v = (input[0].value - input[1].value) * 2
        return (gradient*v/v.size,-gradient*v/v.size)

    @staticmethod
    def _str(input):
        return f"{input[0]._str()}@{input[1]._str()}"

    @staticmethod
    def _latex(input):
        return r""+input[0]._latex()+"x"+input[1]._latex()

def exp(child:Array):
    return Exp.apply(child)

def sin(child:Array):
    return Sin.apply(child)

def cos(child:Array):
    return Cos.apply(child)

def tan(child:Array):
    return Tan.apply(child)

def transpose(child:Array):
    return Transpose.apply(child)

def inv(child:Array):
    return Inv.apply(child)

def matmul(left:Array, right:Array):
    return Matmul.apply(left, right)

def mean_squared_error(output:Array, target:Array):
    return MeanSquaredError.apply(output, target)