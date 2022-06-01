from autodiff import variable
from autodiff.utils import get_vars
import numpy as np
import math

class Function():
    def __init__(self, dim:int, *args):
        self.func = [None] * dim
        self.dim = dim
        vars = []
        for v in args:
            vars.append(v)
        self.vars = vars

    def eval(self, **kwargs):
        """
        evaluate at given environment
        
        param:
            kwargs : variable-values should be given, default 1
        
        return:
            value-vector (1d numpy array)
        """
        res = np.zeros(self.dim)
        env = {}
        for var in self.vars:
            env[var] = kwargs.get(var, 1)
        for i in range(0, self.dim):
            res[i] = self.func[i].eval(**env)
        return res

    def autodiff(self, var:str, **kwargs):
        """
        calculates gradient for var at given envirnment
        
        param:
            var : variable-name to calculate derivative for, e.g. "x"
            kwargs : variable-values should be given, default 1
        
        return:
            gradiant-vector (1d numpy array)
        """
        res = np.zeros(self.dim)
        env = {}
        for var in self.vars:
            env[var] = kwargs.get(var, 1)
        for i in range(0, self.dim):
            res[i] = self.func[i].autodiff(var, **env)
        return res    

    def jacobian(self, **kwargs):
        """
        calculates Jacobian-Matrix at given env

        param:
            kwargs : variable-values should be given, default 1
        
        return:
            jacobian-matrix (2d numpy array)
        """
        env = {}
        for var in self.vars:
            env[var] = kwargs.get(var, 1)
        jac = np.zeros((self.dim, len(self.vars)), dtype="float32")
        for i in range(0, self.dim):
            for j in range(0, len(self.vars)):
                jac[i,j] = self.func[i].autodiff(self.vars[j], **env)
        return jac

    def variance(self, covmat, pos):
        """
        calculates covarianz-matrix for values
        
        param:
            covmat : covariance-matrix for input variables as numpy array
            pos : space-vector (in order of variables, given as 1d numpy array)
        
        return:
            covarainz-matrix (2d numpy array)
        """
        env = {}
        for i in range(0, len(self.vars)):
            env[self.vars[i]] = pos[i]
        jac = self.jacobian(**env)
        return np.matmul(np.matmul(jac, covmat), np.transpose(jac))

    def covar_from_val(self, values):
        """
        calculates covarianz-matrix for values
        
        param:
            value : 2d numpy array, rows correspond to one dataset, columns in order of variables
        
        return:
            covarainz-matrix (2d numpy array)
        """
        cov = cross_covariance(values)
        env = {}
        m = np.sum(values, axis=0) / values.shape[0]
        for i in range(0, len(self.vars)):
            env[self.vars[i]] = m[i]
        jac = self.jacobian(**env)
        return np.matmul(np.matmul(jac, cov), np.transpose(jac))

    def vars(self):
        return self.vars

    def __getitem__(self, key):
        if key < 0 or key > self.dim:
            raise ValueError("index out of range")
        return self.func[key]

    def __setitem__(self, key, value):
        if key < 0 or key > self.dim:
            raise ValueError("index out of range")
        for var in get_vars(value):
            if var not in self.vars:
                raise ValueError("variale mismatch")
        self.func[key] = value

    def __str__(self):
        s = "[ " + self.func[0].__str__() + "\n"
        for i in range(1, self.dim-1):
            s += "  " + self.func[i].__str__() + "\n"
        s += "  " + self.func[self.dim-1].__str__() + " ]"
        return s

def cross_covariance(valuematrix):
    """
    calculates cross covariance matrix from value matrix

    param:
        valuematrix: rows correspond to datasets, columns to variables

    returns:
        covariance matrix as numpy array
    """
    s = valuematrix.shape
    if len(s) > 2 or len(s) < 0:
        raise ValueError("wrong value_matrix format")
    m = np.sum(valuematrix, axis=0) / s[0]
    v = valuematrix - m
    cov = np.matmul(np.transpose(v), v)
    cov = cov / s[0]
    return cov

def correlation_matrix(covarmat):
    """
    calculates correlation matrix from covariance matrix

    param:
        covarmat: covariance matrix as numpy array

    returns:
        correlation as numpy array
    """
    size = covarmat.shape
    cor = np.zeros(size)
    for i in range(0, size[0]):
        for j in range(0, size[0]):
            cor[i,j] = covarmat[i,j] / (math.sqrt(covarmat[i,i])* math.sqrt(covarmat[j,j]))
    return cor