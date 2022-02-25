import math
import cmath
from symbolic_math.utils import *

"""
TODO:
implement all symbols from 
https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols
"""

greek_names = [
    "alpha",
    "beta",
    "gamma",
    "Gamma", 
    "theta",
    "Theta",
    "vartheta", 
    "delta",
    "Delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "kappa",
    "lambda",
    "Lambda",
    "mu",
    "nu",
    "xi",
    "Xi",
    "Pi",
    "rho",
    "varrho",
    "sigma",
    "Sigma",
    "tau",
    "upsilon",
    "Upsilon",
    "varphi",
    "Phi",
    "chi",
    "psi",
    "Psi",
    "omega",
    "Omega"
]

constants = {
    "pi" : math.pi,
    "infty" : math.inf,
    "phi" : ( 1 + math.sqrt(5) ) / 2
}

def get_greek_symbol_list():
    greek_symbols = [ Symbol(n, "\\" + n) for n in greek_names ]
    return greek_symbols

def objectify_input(inp):
    out = inp
    if is_number(inp):
        out = Number(inp)

    elif is_string(inp):
        out = symbol_constructor(inp)

    return out


def symbol_constructor(name):
    if name in constants:
        return Constant(name, constants[name], "\\"+name)
    elif name in greek_names:
        return Symbol(name, "\\"+name)
    else:
        return Symbol(name)


class MathObj(object):
    def __init__(self):
        pass

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def latex(self):
        raise NotImplementedError



class Symbol(MathObj):
    num_args = 0

    def __init__(self, name, latex_str = None):
        super(Symbol, self).__init__()
        self.name = name
        if latex_str is None:
            self.latex_str = str(name)
        else:
            self.latex_str = latex_str

    def latex(self):
        return self.latex_str

    def evaluate(self, **kwargs):
        if self.name in kwargs:
            return kwargs[self.name]
        else:
            return self

    def atoms(self, ttype):
        if type(self) == ttype:
            return set([self.name])
        else:
            return set()

    def __str__(self):
        return self.name


class Constant(Symbol):
    def __init__(self, name, value, latex_str):
        super(Constant, self).__init__(name, latex_str)
        self.value = value

    def evaluate(self):
        return self.value


class Number(Symbol):
    def __init__(self, val):
        super(Number, self).__init__(str(val), latex_str = str(val))
        self.value = float(val)

    def evaluate(self, **kwargs):
        return self.value


class Function(MathObj):
    def __init__(self):
        super(Function, self).__init__()

    def atoms(self, ttype):
        raise NotImplementedError



class GenericFunction(Function):
    def __init__(self, name, *args):
        super(GenericFunction, self).__init__()
        self.name = name
        self.num_args = len(args)
        self.arg_list = [objectify_input(a) for a in args]

    @property
    def args(self):
        return self.arg_list

    @args.setter
    def args(self, alist):
        self.arg_list = [objectify_input(a) for a in alist]
        self.num_args = len(alist)

    def atoms(self, ttype):
        rlist = set()
        if ttype == Function or ttype == GenericFunction:
            rlist.update([self.name])

        for arg in self.arg_list:
            rlist.update(arg.atoms(ttype))

        return rlist

    def latex(self):
        ss = "\\operatorname{" + self.name + "} ("
        for arg in self.arg_list:
            ss += arg.latex() + ","

        ss = ss[:-1]
        ss += ")"
        return ss 

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def __str__(self):
        ss = ""
        for arg in self.arg_list:
            ss += arg.__str__() + ", "
        ss = ss[:-2]
        return self.name + "( " + ss + " )"


class BinaryFunction(Function):
    num_args = 2
    def __init__(self, left, right):
        super(BinaryFunction, self).__init__()
        self.left = objectify_input(left)
        self.right = objectify_input(right)

    @property
    def args(self):
        return [self.left, self.right]

    @args.setter
    def args(self, alist):
        self.left, self.right = alist

    def atoms(self, ttype):
        rlist = set()
        if ttype == Function or ttype == BinaryFunction:
            rlist.update([self.name])

        rlist.update(self.left.atoms(ttype))
        rlist.update(self.right.atoms(ttype))
        return rlist

    def __str__(self):
        return self.name + "( " + self.left.__str__() + ", " + self.right.__str__() +" )"


class UnaryFunction(Function):
    num_args = 1
    def __init__(self, arg):
        super(UnaryFunction, self).__init__()
        self.arg = objectify_input(arg)

    @property
    def args(self):
        return [self.arg]

    @args.setter
    def args(self, alist):
        self.arg = alist[0]

    def atoms(self, ttype):
        rlist = set()
        if ttype == Function or ttype == UnaryFunction:
            rlist.update([self.name])

        rlist.update(self.arg.atoms(ttype))
        return rlist

    def __str__(self):
        return self.name + "( " + self.arg.__str__() + " )"


