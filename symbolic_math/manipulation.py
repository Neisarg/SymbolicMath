import math
import cmath
import symbolic_math as sm
from symbolic_math.utils import *


def get_function_by_name(name):
    if name in sm.functions.__dict__:
        klass = sm.functions.__dict__[name]
    elif name in sm.trigonometry.__dict__:
        klass = sm.trigonometry.__dict__[name]
    if type(klass) is type:
        return klass

    raise Exception("func {} not found".format(name))


def subsitute(exp, ffrom, tto):
    node = exp
    if node.name == ffrom:
        if node.num_args > 0:
            if type(tto) == str:
                cls = get_function_by_name(tto)
            elif type(tto) == type:
                cls = tto 
            elif isinstance(tto, sm.MathObj):
                raise Exception("Cannot replace an object for a function") 
            if cls.num_args != node.num_args:
                raise Exception(
                    "{} has {} args whwers as {} has {} args, they should match".format(ffrom, node.num_args, tto,
                                                                                        cls.num_args))
            new_args = []
            for n in node.args:
                new_args.append(subsitute(n, ffrom, tto))

            new_func = cls(*new_args)
            return new_func
        else:
            if type(tto) == str:
                return sm.symbol_constructor(tto)
            elif isinstance(tto, sm.MathObj):
                return tto
            else:
                return sm.Number(tto)
    else:
        if node.num_args > 0:
            new_args = []
            for n in node.args:
                new_args.append(subsitute(n, ffrom, tto))
            node.args = new_args
            return node
        else:
            return node

