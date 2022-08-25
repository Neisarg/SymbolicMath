from symbolic_math.base import *
from symbolic_math.functions import *
from symbolic_math.trigonometry import *
import random
from copy import deepcopy

COMMON_VARS =  ["x", "y", "z", "a", "b", "c", "k", "l", "m", "n"] + greek_names

COMMON_CONSTS = [Number(-1), Constant("pi", math.pi, "\\pi"), Constant("phi", ( 1 + math.sqrt(5) ) / 2, "\\phi")]

inverse_pairs = [
    ("Add", "Subtract"), 
    ("Multiply", "Divide"),
    ("ln", "exp"), 
    ("sin", "asin"),
    ("cos", "acos"),
    ("tan", "atan"),
    ("csc", "acsc"),
    ("sec", "asec"),
    ("cot", "acot"),
    ("sinh", "asinh"),
    ("cosh", "acosh"),
    ("tanh", "atanh")
]

inverse_dict = dict(inverse_pairs)
inverse_dict.update({k:v for v, k in inverse_pairs}) 


def depth(expr):
    if isinstance(expr, sm.Symbol):
        return 1
    else:
        if isinstance(expr, sm.Function):
            return 1 + max([ depth(arg) for arg in expr.args ])

def get_inverse(f):
    if type(f) == str:
        fname = f
    else:
        fname = f.name

    if fname in inverse_dict:
        return sm.__dict__[inverse_dict[fname]]
    else:
        return None


def random_rational(entropy = 2):
    numerator = random.choice(range(1, 10**entropy))
    denominator = random.choice(range(2, 10**entropy))
    return sm.Divide(sm.Number(numerator), sm.Number(denominator))


def random_pos_integer(entropy=2):
    num = random.choice(range(2, 10 ** entropy))
    return sm.Number(num)


def random_number(entropy=2):
    if random.choice([True, False]):
        return random_rational(entropy)
    else:
        return random_pos_integer(entropy)

# def ops_on_a_side(eq, item, side = "both", ops = "Add"):
#     left = eq.left
#     right = eq.right
#     klass = sm.__dict__[ops]
#     if side == "left":
#         left = klass(left, item)
#     elif side == "right":
#         right = klass(right, item)
#     elif side == "both":
#         left = klass(left, item)
#         right = klass(right, item)

#     return sm.EQ(left, right)


# def get_subsitution_list(eq, domains, only_var = True):
#     if only_var:
#         eq_vars = list(eq.atoms(sm.Symbol))
#     else:
#         eq_vars = list(eq.atoms(sm.Symbol)) + list(eq.atoms(sm.Number)) + list(eq.atoms(sm.Constant))

#     funclist = []
#     for d in domains:
#         funclist.extend(functions[d])

#     varlist = deepcopy(variables)
#     for v in eq_vars:
#         if v in varlist:
#             varlist.remove(v)

#     if len(eq_vars) == 0:
#         return None
#     sub_dict = {}
#     for v in eq_vars:
#         if random.choice([True, False]):
#             ttype = random.choice(["r", "v", "f"])
#             if ttype == "r":
#                 item = random_number()
#             elif ttype == "v":
#                 item = random.choice(variables + constants)
#             elif ttype == "f":
#                 item = grow_random_func_by_domain(funclist, varlist, max_depth=2)

#             sub_dict[v] = item
#     if len(sub_dict) == 0:
#         return None
#     else:
#         return sub_dict


# def get_function_subsitution_list(eq, domains):
#     unary_flist = eq.atoms(sm.UnaryFunction)
#     binary_flist = eq.atoms(sm.BinaryFunction)

#     if "EQ" in binary_flist:
#         binary_flist.remove("EQ")

#     funclist = []
#     for d in domains:
#         funclist.extend(functions[d])

#     unary_pool = list(filter(lambda f: f.num_args == 1, funclist))
#     binary_pool = list(filter(lambda f: f.num_args == 2, funclist))

#     sub_dict = {}

#     for f in unary_flist:
#         if random.choice([True, False]):
#             sub_dict[f] = random.choice(unary_pool)

#     for f in binary_flist:
#         if random.choice([True, False]):
#             sub_dict[f] = random.choice(binary_pool)

#     return sub_dict


# def subsitute_on_a_side(eq, sub_dict, side = "both"):
#     left = deepcopy(eq.left)
#     right = deepcopy(eq.right)

#     if side == "left":
#         for v, item in sub_dict.items():
#             left = sm.subsitute(left, v, item)
#     elif side == "right":
#         for v, item in sub_dict.items():
#             right = sm.subsitute(right, v, item)

#     elif side == "both":
#         for v, item in sub_dict.items():
#             left = sm.subsitute(left, v, item)
#             right = sm.subsitute(right, v, item)

#     return sm.EQ(left, right)


def grow_random_func_by_domain(funclist, varlist, max_depth = 3, curr_depth = 0):
    func = random.choice(funclist)
    if curr_depth == max_depth:
        args = []
        num_done = False
        for ii in range(func.num_args):
            if not num_done:
                ttype = random.choice(["r", "v"])
            else:
                ttype = "v"

            if ttype == "r":
                args.append(random_number())
                num_done = True
            elif ttype == "v":
                args.append(deepcopy(random.choice(varlist)))

        return func(*args)

    else:
        args = []
        for ii in range(func.num_args):
            args.append(grow_random_func_by_domain(funclist, varlist, max_depth, curr_depth+1))

        return func(*args)


# def binary_ops_expressions(eq1, eq2, ops = "Add"):
#     klass = sm.__dict__[ops]
#     assert klass.num_args == 2
#     return sm.EQ(klass(eq1.left, eq2.left), klass(eq1.right, eq2.right))


# def binary_ops_expressions_sided(eq1, eq2, ops = "Add"):
#     klass = sm.__dict__[ops]
#     assert klass.num_args == 2
#     if random.choice([True, False]):
#         if random.choice([True, False]):
#             return sm.EQ(klass(eq1.left, eq2.left), eq1.right)
#         else:
#             return sm.EQ(klass(eq1.left, eq2.right), eq1.right)
#     else:
#         if random.choice([True, False]):
#             return sm.EQ(eq1.left, klass(eq1.right, eq2.right))
#         else:
#             return sm.EQ(eq1.left, klass(eq1.right, eq2.left))


# def unary_ops_expressions(eq, ops, side):
#     if type(ops) == str:
#         klass = sm.__dict__[ops]
#     else:
#         klass = ops
#     assert klass.num_args == 1
#     if side == "left":
#         return sm.EQ(klass(eq.left), eq.right)
#     if side == "right":
#         return sm.EQ(eq.left, klass(eq.right))
#     if side == "both":
#         return sm.EQ(klass(eq.left), klass(eq.right))


# def move_term_left_to_right(eq):
#     if isinstance(eq.left, sm.Function):
#         klass = type(eq.left) 
#         inv_klass = get_inverse(klass)
#         if inv_klass is not None:
#             if klass.num_args == 2:
#                 neq = sm.EQ(eq.left.left, inv_klass(eq.right, eq.left.right))
#                 return neq
#             elif klass.num_args == 1:
#                 neq = sm.EQ(eq.left.arg, inv_klass(eq.right))
#             else:
#                 raise Exception("only unary and binary functions handeled")
#         else:
#             if klass.name == "Pow":
#                 expr, r = eq.left.left, eq.left.right
#                 new_right = sm.Pow(eq.right, sm.Divide(1, r))
#                 neq = sm.EQ(expr, new_right)
#                 return neq

#             elif klass.name == "log":
#                 value = eq.left.arg
#                 new_right = sm.Pow(sm.Number(10), eq.right)
#                 neq = sm.EQ(value, new_right)
#                 return neq

#             elif klass.name == "log_b":
#                 value, base = eq.left.left, eq.left.right
#                 new_right = sm.Pow(base, eq.right)
#                 neq = sm.EQ(value, new_right)
#                 return neq
#     elif isinstance(eq.left, sm.Number):
#         if eq.left.value == 0:
#             return None
#         else:
#             return sm.EQ(sm.Number(0), sm.Subtract(eq.right, eq.left))

#     elif isinstance(eq.left, sm.Symbol):
#         return sm.EQ(sm.Number(0), sm.Subtract(eq.right, eq.left))

#     return None


# def move_term_right_to_left(eq):
#     if isinstance(eq.right, sm.Function):
#         klass = type(eq.right) 
#         inv_klass = get_inverse(klass)
#         if inv_klass is not None:
#             if klass.num_args == 2:
#                 neq = sm.EQ(inv_klass(eq.left, eq.right.right), eq.right.left)
#                 return neq
#             elif klass.num_args == 1:
#                 neq = sm.EQ(inv_klass(eq.left), eq.right.arg)
#             else:
#                 raise Exception("only unary and binary functions handeled")
#         else:
#             if klass.name == "Pow":
#                 expr, r = eq.right.left, eq.right.right
#                 new_left = sm.Pow(eq.left, sm.Divide(1, r))
#                 neq = sm.EQ(new_left, expr)
#                 return neq

#             elif klass.name == "log":
#                 value = eq.right.arg
#                 new_left = sm.Pow(sm.Number(10), eq.left)
#                 neq = sm.EQ(new_left, value)
#                 return neq

#             elif klass.name == "log_b":
#                 value, base = eq.right.left, eq.right.right
#                 new_left = sm.Pow(base, eq.left)
#                 neq = sm.EQ(new_left, value)
#                 return neq
#             elif klass.name == "Negative":
#                 new_left = sm.Add(eq.left, eq.right.arg)
#                 neq = sm.EQ(new_left, sm.Number(0))
#                 return neq
    
#     elif isinstance(eq.right, sm.Number):
#         if eq.right.value == 0:
#             return None
#         else:
#             return sm.EQ(sm.Subtract(eq.left, eq.right), sm.Number(0))

#     elif isinstance(eq.right, sm.Symbol):
#         return sm.EQ(sm.Subtract(eq.left, eq.right), sm.Number(0))

#     return None


# def swap_left_right(eq):
#     return sm.EQ(eq.right, eq.left)


# def move_all_to_left(eq):
#     return sm.EQ(sm.Subtract(eq.left, eq.right), sm.Number(0))


# def move_all_to_right(eq):
#     return sm.EQ(sm.Number(0), sm.Subtract(eq.right, eq.left))
