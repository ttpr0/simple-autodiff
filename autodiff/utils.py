from autodiff import *

def get_vars(func:Expr):
    for node in func.input:
        if type(node) == Variable:
            yield node.name
        else:
            yield from get_vars(node)

def reset_grads(expr:Expr):
    expr.gradient = 0
    for node in expr.input:
        reset_grads(node)

def simplify(func:Expr) -> Expr:
    inputs = []
    for node in func.input:
        inputs.append(simplify(node))
    func.input = tuple(inputs)
    if type(func) == Add:
        func = _simplify_add(func)
    if type(func) == Sub:
        func = _simplify_sub(func)
    if type(func) == Multiply:
        func = _simplify_multiply(func)
    if type(func) == Divide:
        func = _simplify_divide(func)
    if type(func) == Pow:
        func = _simplify_pow(func)
    return func
    
def _simplify_add(node:Add):
    l = node.input[0]
    r = node.input[1]
    if type(l) == Const and type(r) == Const:
        node = Const(l.eval() + r.eval())
    if type(l) == Const and l.eval() == 0:
        node = r
    if type(r) == Const and r.eval() == 0:
        node = l
    if type(l) == Const:
        if type(r) == Add:
            if type(r.input[0]) == Const:
                node = Add(Const(l.eval()+r.input[0].eval()), r.input[1])
            if type(r.input[1]) == Const:
                node = Add(Const(l.eval()+r.input[1].eval()), r.input[0])
        if type(r) == Sub:
            if type(r.input[0]) == Const:
                node = Sub(Const(l.eval()+r.input[0].eval()), r.input[1])
            if type(r.input[1]) == Const:
                node = Add(Const(l.eval()-r.input[1].eval()), r.input[0])
    if type(r) == Const:
        if type(l) == Add:
            if type(l.input[0]) == Const:
                node = Add(Const(r.eval()+l.input[0].eval()), l.input[1])
            if type(l.input[1]) == Const:
                node = Add(Const(r.eval()+l.input[1].eval()), l.input[0])
        if type(l) == Sub:
            if type(l.input[0]) == Const:
                node = Sub(Const(r.eval()+l.input[0].eval()), l.input[1])
            if type(l.input[1]) == Const:
                node = Add(Const(l.eval()-l.input[0].eval()), r.input[1])
    return node

def _simplify_sub(node:Sub):
    l = node.input[0]
    r = node.input[1]
    if type(l) == Const and type(r) == Const:
        node = Const(l.eval() - r.eval())
    if type(l) == Const and l.eval() == 0:
        node = Const(-r.eval())
    if type(r) == Const and r.eval() == 0:
        node = l
    if type(l) == Const:
        if type(r) == Add:
            if type(r.input[0]) == Const:
                node = Sub(Const(l.eval()-r.input[0].eval()), r.input[1])
            if type(r.input[1]) == Const:
                node = Sub(Const(l.eval()-r.input[1].eval()), r.input[0])
        if type(r) == Sub:
            if type(r.input[0]) == Const:
                node = Add(Const(l.eval()-r.input[0].eval()), r.input[1])
            if type(r.input[1]) == Const:
                node = Sub(Const(l.eval()+r.input[0].eval()), r.input[1])
    if type(r) == Const:
        if type(l) == Add:
            if type(l.input[0]) == Const:
                node = Sub(Const(r.eval()-l.input[0].eval()), l.input[1])
            if type(l.input[1]) == Const:
                node = Sub(Const(r.eval()-l.input[1].eval()), l.input[0])
        if type(l) == Sub:
            if type(l.input[0]) == Const:
                node = Add(Const(r.eval()-l.input[0].eval()), l.input[1])
            if type(l.input[1]) == Const:
                node = Sub(Const(l.eval()+l.input[1].eval()), r.input[1])
    return node

def _simplify_multiply(node:Multiply):
    l = node.input[0]
    r = node.input[1]
    if type(l) == Const and type(r) == Const:
        node = Const(l.eval() * l.eval())
    if type(l) == Const and l.eval() == 0:
        node = Const(0)
    if type(r) == Const and r.eval() == 0:
        node = Const(0)
    if type(l) == Const:
        if type(r) == Multiply:
            if type(r.input[0]) == Const:
                node = Multiply(Const(l.eval()*r.input[0].eval()), r.input[1])
            if type(r.input[1]) == Const:
                node = Multiply(Const(l.eval()*r.input[1].eval()), r.input[0])
    if type(r) == Const:
        if type(l) == Multiply:
            if type(l.input[0]) == Const:
                node = Multiply(Const(r.eval()*l.input[0].eval()), l.input[1])
            if type(l.input[1]) == Const:
                node = Multiply(Const(r.eval()*l.input[1].eval()), l.input[0])
    return node

def _simplify_divide(node:Divide):
    l = node.input[0]
    r = node.input[1]
    if type(l) == Const and type(r) == Const:
        node = Const(l.eval() / r.eval())
    if type(l) == Const and l.eval() == 0:
        node = Const(0)
    return node

def _simplify_pow(node:Pow):
    b = node.input[0]
    e = node.input[1]
    if type(b) == Const:
        node = Const(math.pow(b.eval(), e.eval()))
    if e.eval() == 1:
        node = b
    if e.eval() == 0:
        node = Const(1)
    return node