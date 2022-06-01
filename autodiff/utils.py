import autodiff.array as array

def get_vars(func:array.Array):
    for node in (func.input or []):
        if node.name != None:
            yield node.name
        else:
            yield from get_vars(node)

def reset_grads(expr:array.Array):
    expr._gradient = None
    for node in (expr.input or []):
        reset_grads(node)

def apply_grads(expr:array.Array, lr:float=0.01):
    if expr.operation == None and expr.track_grads:
        expr.value = expr.value - (expr.gradient*lr)
    for node in (expr.input or []):
        if type(node) == array.Array:
            apply_grads(node, lr) 

# def simplify(func:Expr) -> Expr:
#     inputs = []
#     for node in func.input:
#         inputs.append(simplify(node))
#     func.input = tuple(inputs)
#     if type(func) == Add:
#         func = _simplify_add(func)
#     if type(func) == Sub:
#         func = _simplify_sub(func)
#     if type(func) == Multiply:
#         func = _simplify_multiply(func)
#     if type(func) == Divide:
#         func = _simplify_divide(func)
#     if type(func) == Pow:
#         func = _simplify_pow(func)
#     return func
    
# def _simplify_add(node:Add):
#     l = node.input[0]
#     r = node.input[1]
#     if type(l) == Param and type(r) == Param:
#         node = Param(l.eval() + r.eval())
#     if type(l) == Param and l.eval() == 0:
#         node = r
#     if type(r) == Param and r.eval() == 0:
#         node = l
#     if type(l) == Param:
#         if type(r) == Add:
#             if type(r.input[0]) == Param:
#                 node = Add(Param(l.eval()+r.input[0].eval()), r.input[1])
#             if type(r.input[1]) == Param:
#                 node = Add(Param(l.eval()+r.input[1].eval()), r.input[0])
#         if type(r) == Sub:
#             if type(r.input[0]) == Param:
#                 node = Sub(Param(l.eval()+r.input[0].eval()), r.input[1])
#             if type(r.input[1]) == Param:
#                 node = Add(Param(l.eval()-r.input[1].eval()), r.input[0])
#     if type(r) == Param:
#         if type(l) == Add:
#             if type(l.input[0]) == Param:
#                 node = Add(Param(r.eval()+l.input[0].eval()), l.input[1])
#             if type(l.input[1]) == Param:
#                 node = Add(Param(r.eval()+l.input[1].eval()), l.input[0])
#         if type(l) == Sub:
#             if type(l.input[0]) == Param:
#                 node = Sub(Param(r.eval()+l.input[0].eval()), l.input[1])
#             if type(l.input[1]) == Param:
#                 node = Add(Param(l.eval()-l.input[0].eval()), r.input[1])
#     return node

# def _simplify_sub(node:Sub):
#     l = node.input[0]
#     r = node.input[1]
#     if type(l) == Param and type(r) == Param:
#         node = Param(l.eval() - r.eval())
#     if type(l) == Param and l.eval() == 0:
#         node = Param(-r.eval())
#     if type(r) == Param and r.eval() == 0:
#         node = l
#     if type(l) == Param:
#         if type(r) == Add:
#             if type(r.input[0]) == Param:
#                 node = Sub(Param(l.eval()-r.input[0].eval()), r.input[1])
#             if type(r.input[1]) == Param:
#                 node = Sub(Param(l.eval()-r.input[1].eval()), r.input[0])
#         if type(r) == Sub:
#             if type(r.input[0]) == Param:
#                 node = Add(Param(l.eval()-r.input[0].eval()), r.input[1])
#             if type(r.input[1]) == Param:
#                 node = Sub(Param(l.eval()+r.input[0].eval()), r.input[1])
#     if type(r) == Param:
#         if type(l) == Add:
#             if type(l.input[0]) == Param:
#                 node = Sub(Param(r.eval()-l.input[0].eval()), l.input[1])
#             if type(l.input[1]) == Param:
#                 node = Sub(Param(r.eval()-l.input[1].eval()), l.input[0])
#         if type(l) == Sub:
#             if type(l.input[0]) == Param:
#                 node = Add(Param(r.eval()-l.input[0].eval()), l.input[1])
#             if type(l.input[1]) == Param:
#                 node = Sub(Param(l.eval()+l.input[1].eval()), r.input[1])
#     return node

# def _simplify_multiply(node:Multiply):
#     l = node.input[0]
#     r = node.input[1]
#     if type(l) == Param and type(r) == Param:
#         node = Param(l.eval() * l.eval())
#     if type(l) == Param and l.eval() == 0:
#         node = Param(0)
#     if type(r) == Param and r.eval() == 0:
#         node = Param(0)
#     if type(l) == Param:
#         if type(r) == Multiply:
#             if type(r.input[0]) == Param:
#                 node = Multiply(Param(l.eval()*r.input[0].eval()), r.input[1])
#             if type(r.input[1]) == Param:
#                 node = Multiply(Param(l.eval()*r.input[1].eval()), r.input[0])
#     if type(r) == Param:
#         if type(l) == Multiply:
#             if type(l.input[0]) == Param:
#                 node = Multiply(Param(r.eval()*l.input[0].eval()), l.input[1])
#             if type(l.input[1]) == Param:
#                 node = Multiply(Param(r.eval()*l.input[1].eval()), l.input[0])
#     return node

# def _simplify_divide(node:Divide):
#     l = node.input[0]
#     r = node.input[1]
#     if type(l) == Param and type(r) == Param:
#         node = Param(l.eval() / r.eval())
#     if type(l) == Param and l.eval() == 0:
#         node = Param(0)
#     return node

# def _simplify_pow(node:Pow):
#     b = node.input[0]
#     e = node.input[1]
#     if type(b) == Param:
#         node = Param(b.eval()**e.eval())
#     if e.eval() == 1:
#         node = b
#     if e.eval() == 0:
#         node = Param(1)
#     return node