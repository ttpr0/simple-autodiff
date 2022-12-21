from autodiff.array import Array
from abc import abstractmethod
import numpy as np

TRACK_COMP = False

class track_computation:
    def __init__(self):
        pass
    def __enter__(self):
        global TRACK_COMP
        TRACK_COMP = True
    def __exit__(self, type, value, traceback):
        global TRACK_COMP
        TRACK_COMP = False

class Operation():

    @classmethod
    def apply(cls, *input):
        """
        applies a operation on given input
        
        Args:
            input: input arguments (usually Arrays)
        
        Returns:
            the computed output Array
        """
        input_ = tuple(item.value if type(item) == Array else item for item in input)
        cls._validate_input(input_)
        value, params = cls._eval(input_)
        if any(i.track_grads for i in input if type(i) == Array):
            if TRACK_COMP:
                arr = Array(value, track_grads=True)
                arr.operation = cls
                arr.input = tuple(input)
                arr.params = params
                return arr
            else:
                return Array(value, track_grads=True)
        else:
            return Array(value, track_grads=False)

    @staticmethod
    @abstractmethod
    def _validate_input(input):
        """
        validates the input arguments for the operation
        
        Args:
            input: input arguments to operation (Arrays are given as their underlying numpy array)
        
        Throws:
            on invalid input
        """
        pass

    @staticmethod
    @abstractmethod
    def _eval(input):
        """
        evaluates operation for given input
        
        Args:
            input: operation input (Arrays are given as their underlying numpy array)
        
        Returns:
            result: the operation result as a numpy array
            params: a object containing parameters to be used in backward pass
        """
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
    def _backward(gradient, input, params):
        """
        computes backward pass for operation with respect to inputs
        
        Args:
            gradient: numpy array containing radient for current Array
            input: operation input (Arrays are given as their underlying numpy array)
            params: parameter object returned from _eval method
        
        Returns:
            a tuple containing gradient with respect to every input as numpy arrays
        """
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
        if input[0].shape != input[1].shape:
            raise ValueError("dimensions do not match")

    @staticmethod
    def _eval(input):
        return input[0] + input[1], None

    @staticmethod
    def _diff(input, gradient):
        # return Add(self.input[0].diff(var), self.input[1].diff(var))
        pass

    @staticmethod
    def _forward(input):
        # return self.input[0].forward(var) + self.input[1].forward(var)
        pass

    @staticmethod
    def _backward(gradient, input, params):
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
        if input[0].shape != input[1].shape:
            raise ValueError("dimensions do not match")

    @staticmethod
    def _eval(input):
        return input[0] - input[1], None

    @staticmethod
    def _diff(input, gradient):
        # return Sub(self.input[0].diff(var), self.input[1].diff(var))
        pass

    @staticmethod
    def _forward(input):
        # return self.input[0].forward(var) - self.input[1].forward(var)
        pass

    @staticmethod
    def _backward(gradient, input, params):
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
        if input[0].shape != input[1].shape:
            raise ValueError("dimensions do not match")

    @staticmethod
    def _eval(input):
        return input[0] * input[1], None

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
    def _backward(gradient, input, params):
        return (gradient*input[1],gradient*input[0])

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
        if input[0].shape != input[1].shape:
            raise ValueError("dimensions do not match")

    @staticmethod
    def _eval(input):
        return input[0] / input[1], None

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
    def _backward(gradient, input, params):
        return (gradient/input[1],-gradient*input[0]/input[1]**2)

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
        if input[0].shape != input[1].shape:
            raise ValueError("dimensions do not match")

    @staticmethod
    def _eval(input):
        return input[0] ** input[1], None

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
    def _backward(gradient, input, params):
        db = input[1] * input[0]**(input[1]-1)
        de = np.log(input[0]) * input[0]**input[1]
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
        return np.log(input[0]), None

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
    def _backward(gradient, input, params):
        return (gradient/input[0],)

    @staticmethod
    def _str(input):
        return f"ln({input[0]._str()})"

    @staticmethod
    def _latex(input):
        return r"ln("+input[0]._latex()+")"


class Expand(Operation):
    @staticmethod
    def _validate_input(input):
        if input[0].shape != (1,):
            raise ValueError("only scalar can be expanded")
        if type(input[1]) != tuple:
            raise ValueError("dimension not valid")

    @staticmethod
    def _eval(input):
        return np.full(input[1], input[0]), None

    @staticmethod
    def _diff(input, gradient):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input, params):
        return (np.array(np.sum(gradient)),)

    @staticmethod
    def _str(input):
        return f"{input[0]._str()}"
    
    @staticmethod
    def _latex(input):
        return f"{input[0]._latex()}"


class Exp(Operation):
    @staticmethod
    def _validate_input(input):
        return

    @staticmethod
    def _eval(input):
        return np.exp(input[0]), None

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
    def _backward(gradient, input, params):
        return (gradient*np.exp(input[0]),)

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
        return np.sin(input[0]), None

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
    def _backward(gradient, input, params):
        return (gradient*np.cos(input[0]),)

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
        return np.cos(input[0]), None

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
    def _backward(gradient, input, params):
        return (-gradient*np.sin(input[0]),)

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
        return np.tan(input[0]), None

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
    def _backward(gradient, input, params):
        return (gradient/np.cos(input[0])**2,)

    @staticmethod
    def _str(input):
        return f"tan({input[0]._str()})"

    @staticmethod
    def _latex(input):
        return r"tan("+input[0]._latex()+")"


class MeanSquaredError(Operation):
    @staticmethod
    def _validate_input(input):
        if input[0].shape != input[1].shape:
            raise ValueError("target dimension must match output dimension")
        return

    @staticmethod
    def _eval(input):
        v = (input[0] - input[1])**2
        return np.array([np.sum(v)/v.size]), None

    @staticmethod
    def _diff(input, gradient):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input, params):
        v = (input[0] - input[1]) * 2
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
        return np.transpose(input[0]), None

    @staticmethod
    def _diff(input):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input, params):
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
        if len(input[0].shape) < 2 or input[0].shape[-1] != input[0].shape[-2]:
            raise ValueError("inv only allowed for quadratic matricies")
        return

    @staticmethod
    def _eval(input):
        return np.linalg.inv(input[0]), None

    @staticmethod
    def _diff(input, gradient):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input, params):
        inv = np.linalg.inv(input[0])
        *a, b, c = tuple(i for i in range(len(input[0].shape)))
        t = np.transpose(inv, (*a,c,b))
        v = -np.matmul(t, inv)
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
        if len(input[0].shape) < 2 or len(input[1].shape) < 2:
            raise ValueError("matmul only for 2d matricies")
        if input[0].shape[-2] != input[1].shape[-1]:
            raise ValueError("dimesion error")
        return

    @staticmethod
    def _eval(input):
        return np.matmul(input[0], input[1]), None

    @staticmethod
    def _diff(input, gradient):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input, params):
        *a, b, c = tuple(i for i in range(len(input[0].shape)))
        in0_t = np.transpose(input[0], (*a,c,b))
        *a, b, c = tuple(i for i in range(len(input[1].shape)))
        in1_t = np.transpose(input[1], (*a,c,b))
        return (np.matmul(gradient, in1_t), np.matmul(in0_t, gradient))

    @staticmethod
    def _str(input):
        return f"{input[0]._str()}@{input[1]._str()}"

    @staticmethod
    def _latex(input):
        return r""+input[0]._latex()+"x"+input[1]._latex()


class Reshape(Operation):
    @staticmethod
    def _validate_input(input):
        if type(input[0]) != np.ndarray:
            raise ValueError("only Arrays can be reshaped")
        if type(input[1]) != tuple:
            raise ValueError("dimension not valid")

    @staticmethod
    def _eval(input):
        return np.reshape(input[0], input[1]), input[0].shape

    @staticmethod
    def _diff(input, gradient):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input, params):
        return (np.reshape(gradient, params),)

    @staticmethod
    def _str(input):
        return f"{input[0]._str()}"
    
    @staticmethod
    def _latex(input):
        return f"{input[0]._latex()}"


class Conv2D(Operation):
    @staticmethod
    def _validate_input(input):
        if len(input[0].shape) == 3 and len(input[1].shape) == 4:
            if input[0].shape[0] <= input[1].shape[1] or input[0].shape[1] <= input[1].shape[2]:
                raise ValueError("in1 must be larger than in2")
            if input[0].shape[2] != input[1].shape[3]:
                raise ValueError("invalid input dimensions")
        else:
            raise ValueError("invalid input dimensions")

        if input[1].shape[1] % 2 != 1 or input[1].shape[2] % 2 != 1:
            raise ValueError("filter has to have an odd dimension")
        return

    @staticmethod
    def _eval(input):
        in_arr = input[0]
        in_kern = input[1]
        out_arr = np.zeros((in_arr.shape[0]-in_kern.shape[1]+1, in_arr.shape[1]-in_kern.shape[1]+1, in_kern.shape[0]))
        for i in range(0, out_arr.shape[2]):
            for j in range(0, in_arr.shape[2]):
                out_arr[:,:,i] += _conv2D(in_arr[:,:,j], in_kern[i,:,:,j])
        return out_arr, None

    @staticmethod
    def _diff(input, gradient):
        pass

    @staticmethod
    def _forward(input):
        pass

    @staticmethod
    def _backward(gradient, input, params):
        in_arr = input[0]
        in_kern = input[1]
        kern_grad = np.zeros(in_kern.shape, dtype=np.float32)
        arr_grad = np.zeros(in_arr.shape, dtype=np.float32)
        for i in range(0, gradient.shape[2]):
            for j in range(0, arr_grad.shape[2]):
                # kernel gradient
                kern_grad[i,:,:,j] = _conv2D(in_arr[:,:,j], gradient[:,:,i])
                # array gradient
                kern_rot = np.rot90(in_kern[i,:,:,j], 2)
                arr_grad[:,:,j] += _conv2D_full(gradient[:,:,i], kern_rot)
        return (arr_grad, kern_grad)

    @staticmethod
    def _str(input):
        return f"conv2d({input[0]._str()})"

    @staticmethod
    def _latex(input):
        return r"conv2d("+input[0]._latex()+")"


def ln(child:Array):
    return Ln.apply(child)

def expand(child:Array, dimension:tuple):
    return Expand.apply(child, dimension)

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

def reshape(child:Array, new_shape:tuple):
    return Reshape.apply(child, new_shape)

def conv2D(arr:Array, kernel:Array):
    return Conv2D.apply(arr, kernel)


def _conv2D(arr:np.ndarray, kern:np.ndarray):
    res_shape = (arr.shape[0]-kern.shape[0]+1, arr.shape[1]-kern.shape[1]+1)
    res_arr = np.zeros(res_shape)
    for i in range(0, res_shape[0]):
            for j in range(0, res_shape[1]):
                res_arr[i,j] = np.sum(arr[i:kern.shape[0]+i,j:kern.shape[1]+j]*kern)
    return res_arr

def _conv2D_full(arr:np.ndarray, kern:np.ndarray):
    pad_0 = kern.shape[0]-1
    pad_1 = kern.shape[1]-1
    res_shape = (arr.shape[0]+pad_0, arr.shape[1]+pad_1)
    res_arr = np.zeros(res_shape)
    for i in range(0, res_shape[0]):
            for j in range(0, res_shape[1]):
                kern_0_min = max(0, pad_0-i)
                kern_1_min = max(0, pad_1-j)
                kern_0_max = min(pad_0+1, res_shape[0]-i)
                kern_1_max = min(pad_1+1, res_shape[1]-j)
                kern_part = kern[kern_0_min:kern_0_max, kern_1_min:kern_1_max]
                arr_0_min = max(0, i-pad_0)
                arr_1_min = max(0, j-pad_1)
                arr_0_max = min(arr.shape[0], i+1)
                arr_1_max = min(arr.shape[1], j+1)
                arr_part = arr[arr_0_min:arr_0_max, arr_1_min:arr_1_max]
                res_arr[i,j] = np.sum(arr_part*kern_part)
    return res_arr