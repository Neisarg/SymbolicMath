import math
import cmath
from symbolic_math.base import *
from symbolic_math.utils import *

class Add(BinaryFunction):
    name = "Add"
    def __init__(self, left, right):
        super(Add, self).__init__(left, right)

    def latex(self):
        return self.left.latex() + "+" + self.right.latex()

    def evaluate(self, **kwargs):
        ls = self.left.evaluate(**kwargs)
        rs = self.right.evaluate(**kwargs)

        if type(ls) in [int, float, complex] and type(rs) in [int, float, complex]:
            return ls + rs
        
        if isinstance(ls, MathObj) and isinstance(rs, MathObj):
            return Add(ls, rs)

        if isinstance(ls, MathObj):
            if rs == 0:
                return ls
            if cmath.isnan(rs) or cmath.isinf(rs):
                return rs

            return Add(ls, rs)

        if ls == 0:
            return rs

        if cmath.isinf(ls) or cmath.isnan(ls):
            return ls

        return Add(ls, rs)


class Subtract(BinaryFunction):
    name = "Subtract"
    def __init__(self, left, right):
        super(Subtract, self).__init__(left, right)

    def latex(self):
        return self.left.latex() + "-" + self.right.latex()

    def evaluate(self, **kwargs):
        ls = self.left.evaluate(**kwargs)
        rs = self.right.evaluate(**kwargs)

        if type(ls) in [int, float, complex] and type(rs) in [int, float, complex]:
            return ls - rs
        
        if isinstance(ls, MathObj) and isinstance(rs, MathObj):
            return Subtract(ls, rs)

        if isinstance(ls, MathObj):
            if rs == 0:
                return ls
            if cmath.isnan(rs):
                return cmath.nan
            if cmath.isinf(rs):
                return -cmath.inf

            return Subtract(ls, rs)

        if ls == 0:
            return Negative(rs)

        if cmath.isinf(ls) or cmath.isnan(ls):
            return ls

        return Subtract(ls, rs)


class Multiply(BinaryFunction):
    name = "Multiply"
    def __init__(self, left, right):
        super(Multiply, self).__init__(left, right)

    def latex(self):
        ls = self.left.latex()
        rs = self.right.latex()

        if isinstance(self.left, BinaryFunction):
            ls = "(" + ls + ")"

        if isinstance(self.right, BinaryFunction):
            rs = "(" + rs + ")"

        return ls + "*" + rs

    def evaluate(self, **kwargs):
        ls = self.left.evaluate(**kwargs)
        rs = self.right.evaluate(**kwargs)
        if type(ls) in [int, float, complex] and type(rs) in [int, float, complex]:
            return ls * rs
        
        if isinstance(ls, MathObj) and isinstance(rs, MathObj):
            return Multiply(ls, rs)

        if isinstance(ls, MathObj):
            if rs == 0:
                return 0
            if rs == 1:
                return ls
            if cmath.isnan(rs) or cmath.isinf(rs):
                return rs

            return Multiply(ls, rs)

        if ls == 0:
            return 0

        if ls == 1:
            return rs

        if cmath.isinf(ls) or cmath.isnan(ls):
            return ls

        return Multiply(ls, rs)


class Divide(BinaryFunction):
    name = "Divide"
    def __init__(self, left, right):
        super(Divide, self).__init__(left, right)

    def latex(self):
        ls = self.left.latex()
        rs = self.right.latex()

        return "\\frac{" + ls + "}{" + rs + "}"

    def evaluate(self, **kwargs):
        ls = self.left.evaluate(**kwargs)
        rs = self.right.evaluate(**kwargs)

        if type(ls) in [int, float, complex] and type(rs) in [int, float, complex]:
            if rs == 0:
                return cmath.inf
            return ls / rs
        
        if isinstance(ls, MathObj) and isinstance(rs, MathObj):
            return Divide(ls, rs)

        if isinstance(ls, MathObj):
            if rs == 0:
                return cmath.inf
            if rs == 1:
                return ls
            if cmath.isnan(rs): 
                return cmath.nan
            if cmath.isinf(rs):
                return 0

            return Divide(ls, rs)

        if ls == 0:
            return 0

        if cmath.isinf(ls) or cmath.isnan(ls):
            return ls
     
        return Divide(ls, rs) 


class Pow(BinaryFunction):
    name = "Pow"
    def __init__(self, left, right):
        super(Pow, self).__init__(left, right)

    def latex(self):
        ls = self.left.latex()
        rs = self.right.latex()

        if self.left.name in ["log", "ln",  "log_b" , 
                                "sin", "cos", "tan", "csc", "sec", "cot", 
                                "asin", "acos", "atan", "acsc", "asec", "acot", 
                                "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"]:
            lsplit = ls.split(" ")
            op, arg = lsplit[0], " ".join(lsplit[1:])
            return op + "^{" + rs +"}" + arg

        if isinstance(self.left, BinaryFunction):
            ls = "(" + ls + ")"

        return ls + "^{" + rs + "}"

    def evaluate(self, **kwargs):
        ls = self.left.evaluate(**kwargs)
        rs = self.right.evaluate(**kwargs)
        if type(ls) in [int, float] and type(rs) in [int, float, complex]:
            if ls == 0:
                if type(rs) in [int, float]:
                    if rs < 0:
                        return cmath.nan
                if type(rs) == complex:
                    return cmath.nan
            try:
                res = ls**rs
                return res
            except:
                res = cmath.nan
                return res
        
        if isinstance(ls, MathObj) and isinstance(rs, MathObj):
            return Pow(ls, rs)

        if isinstance(ls, MathObj):
            if rs == 0:
                return 1
            if rs == 1:
                return ls
            if cmath.isnan(rs) or cmath.isinf(rs): 
                return rs

            return Pow(ls, rs)

        if ls == 0:
            return 0

        if cmath.isinf(ls) or cmath.isnan(ls):
            return ls
     
        return Pow(ls, rs) 


class EQ(BinaryFunction):
    num_args = 2
    name = "EQ"

    def __init__(self, left, right):
        super(EQ, self).__init__(left, right)
        self.error = 1e-5

    def latex(self):
        return self.left.latex() + "=" + self.right.latex()

    def evaluate(self, **kwargs):
        ls = self.left.evaluate(**kwargs)
        rs = self.right.evaluate(**kwargs)
        if type(ls) in [int, float, complex] and type(rs) in [int, float, complex]:
            diff = abs(ls - rs)
            return diff < self.error
        
        if isinstance(ls, MathObj) and isinstance(rs, MathObj):
            return EQ(ls, rs)

        if isinstance(ls, MathObj):
            if cmath.isnan(rs): 
                return cmath.nan

            return EQ(ls, rs)

    
        if cmath.isnan(ls):
            return cmath.nan 

        return EQ(ls, rs) 


class GEQ(BinaryFunction):
    name = "GEQ"

    def __init__(self, left, right):
        super(GEQ, self).__init__(left, right)

    def latex(self):
        return self.left.latex() + "\\geq" + self.right.latex()

    def evaluate(self, **kwargs):
        ls = self.left.evaluate(**kwargs)
        rs = self.right.evaluate(**kwargs)
        if type(ls) in [int, float, complex] and type(rs) in [int, float, complex]:
            return ls >= rs
        
        if isinstance(ls, MathObj) and isinstance(rs, MathObj):
            return GEQ(ls, rs)

        if isinstance(ls, MathObj):
            if cmath.isnan(rs): 
                return cmath.nan

            return GEQ(ls, rs)

    
        if cmath.isnan(ls):
            return cmath.nan 

        return GEQ(ls, rs) 


class LEQ(BinaryFunction):
    name = "LEQ"

    def __init__(self, left, right):
        super(LEQ, self).__init__(left, right)

    def latex(self):
        return self.left.latex() + "\\leq" + self.right.latex()

    def evaluate(self, **kwargs):
        ls = self.left.evaluate(**kwargs)
        rs = self.right.evaluate(**kwargs)
        if type(ls) in [int, float, complex] and type(rs) in [int, float, complex]:
            return ls <= rs
        
        if isinstance(ls, MathObj) and isinstance(rs, MathObj):
            return LEQ(ls, rs)

        if isinstance(ls, MathObj):
            if cmath.isnan(rs): 
                return cmath.nan

            return LEQ(ls, rs)

    
        if cmath.isnan(ls):
            return cmath.nan 

        return LEQ(ls, rs) 


class LT(BinaryFunction):
    name = "LT"

    def __init__(self, left, right):
        super(LT, self).__init__(left, right)

    def latex(self):
        return self.left.latex() + "<" + self.right.latex()

    def evaluate(self, **kwargs):
        ls = self.left.evaluate(**kwargs)
        rs = self.right.evaluate(**kwargs)
        if type(ls) in [int, float, complex] and type(rs) in [int, float, complex]:
            return ls < rs
        
        if isinstance(ls, MathObj) and isinstance(rs, MathObj):
            return LT(ls, rs)

        if isinstance(ls, MathObj):
            if cmath.isnan(rs): 
                return cmath.nan

            return LT(ls, rs)

    
        if cmath.isnan(ls):
            return cmath.nan 

        return LT(ls, rs) 



class GT(BinaryFunction):
    name = "GT"

    def __init__(self, left, right):
        super(GT, self).__init__(left, right)

    def latex(self):
        return self.left.latex() + ">" + self.right.latex()

    def evaluate(self, **kwargs):
        ls = self.left.evaluate(**kwargs)
        rs = self.right.evaluate(**kwargs)
        if type(ls) in [int, float, complex] and type(rs) in [int, float, complex]:
            return ls > rs
        
        if isinstance(ls, MathObj) and isinstance(rs, MathObj):
            return GT(ls, rs)

        if isinstance(ls, MathObj):
            if cmath.isnan(rs): 
                return cmath.nan

            return GT(ls, rs)

    
        if cmath.isnan(ls):
            return cmath.nan 

        return GT(ls, rs) 


class Derivative(BinaryFunction):
    name = "Derivative"

    def __init__(self, left, right):
        super(Derivative, self).__init__(left, right)

    def latex(self):
        ls = self.left.latex()
        if isinstance(self.left, BinaryFunction):
            ls = "(" + ls + ")"

        rs = self.right.latex()
        if isinstance(self.right, BinaryFunction):
            rs = "(" + rs + ")"

        return "\\frac{d" + ls + "}{d" + rs + "}"

    def evaluate(self, **kwargs):
        raise NotImplementedError


class Integral(BinaryFunction):
    name = "Integral"

    def __init__(self, left, right):
        super(Integral, self).__init__(left, right)

    def latex(self):
        raise NotImplementedError

    def evaluate(self, **kwargs):
        raise NotImplementedError


class Abs(UnaryFunction):
    name = "Abs"

    def __init__(self, arg):
        super(Abs, self).__init__(arg)

    def latex(self):
        return "|" + self.arg.latex() + "|"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return Abs(av)
        
        return abs(av) 


class exp(UnaryFunction):
    name = "exp"

    def __init__(self, arg):
        super(exp, self).__init__(arg)

    def latex(self):
        return "e^{" + self.arg.latex() + "}"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return exp(av)
        
        return cmath.exp(av) 



class log(UnaryFunction):
    name = "log"

    def __init__(self, arg):
        super(log, self).__init__(arg)

    def latex(self):
        return "\\log (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        arg_v = self.arg.evaluate(**kwargs)

        if isinstance(arg_v, MathObj):
            return log(arg_v)

        if type(arg_v) in [int, float]:
            if arg_v > 0:
                return math.log10(arg_v)
            elif arg_v < 0:
                return cmath.log10(arg_v)
            else:
                return cmath.nan
        else:
            return cmath.log10(arg_v)


class ln(UnaryFunction):
    name = "ln"

    def __init__(self, arg):
        super(ln, self).__init__(arg)

    def latex(self):
        return "\\ln (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        arg_v = self.arg.evaluate(**kwargs)
        if isinstance(arg_v, MathObj):
            return ln(arg_v)

        if type(arg_v) in [int, float]:
            if arg_v > 0:
                return math.log(arg_v)
            elif arg_v < 0:
                return cmath.log(arg_v)
            else:
                return cmath.nan
        else:
            return cmath.log10(arg_v)


class log_b(BinaryFunction):
    name = "log_b"

    def __init__(self, value, base):
        super(log_b, self).__init__(value, base)

    def latex(self):
        value, base = self.left, self.right
        return "\\log_{" + base.latex() +"} (" + value.latex() + ")"

    def evaluate(self, **kwargs):
        value, base = self.left.evaluate(**kwargs), self.right.evaluate(**kwargs)
        if type(value) in [int, float]:
            if value in [cmath.nan, cmath.inf]:
                return cmath.nan

            if value == 0:
                return cmath.nan

            if value == 1:
                return 0

            if type(base) in [int, float]:    
                if base in [0, 1, cmath.nan, cmath.inf]:
                    return cmath.nan
                else:
                    return math.log(value, base)
            elif type(base) == complex:
                return cmath.log(value, base)
            elif isinstance(base, MathObj):
                return log_b(value, base)

        elif type(value) == complex:
            if type(base) in [int, float]:    
                if base in [0, 1, cmath.nan, cmath.inf]:
                    return cmath.nan
                else:
                    return cmath.log(value, base)
            elif type(base) == complex:
                return cmath.log(value, base)
            elif isinstance(base, MathObj):
                return log_b(value, base)

        elif isinstance(value, MathObj):
            if type(base) in [int, float]:    
                if base in [0, 1, cmath.nan, cmath.inf]:
                    return cmath.nan
            return log_b(value, base)


class sqrt(UnaryFunction):
    name = "sqrt"

    def __init__(self, arg):
        super(sqrt, self).__init__(arg)

    def latex(self):
        return "\\sqrt{" + self.arg.latex() + "}"

    def evaluate(self, **kwargs):
        arg_v = self.arg.evaluate(**kwargs)
        if isinstance(arg_v, MathObj):
            return sqrt(arg_v)
        if type(arg_v) == complex:
            return cmath.sqrt(arg_v)
        if arg_v < 0:
            return cmath.sqrt(arg_v)
        else:
            return math.sqrt(arg_v)

        
class root(BinaryFunction):
    name = "root"

    def __init__(self, expr, r):
        super(root, self).__init__(expr, r)

    def latex(self):
        expr = self.left.latex()
        r = self.right.latex()
        return "\\sqrt[" + r + "]{" + expr + "}"

    def evaluate(self, **kwargs):
        expr = self.left.evaluate(**kwargs)
        r = self.right.evaluate(**kwargs)
        if type(r) == complex:
            return cmath.nan

        if type(expr) in [int, float, complex] and type(r) in [int, float]:
            if r == 0:
                return cmath.nan 

            if expr == 0:
                return 0

            return expr**(1.0/r)

        
        if isinstance(expr, MathObj) and isinstance(r, MathObj):
            return root(expr, r)

        if isinstance(expr, MathObj):
            if r == 0:
                return cmath.nan
            if r == 1:
                return expr
            if cmath.isnan(r): 
                return cmath.nan
            if cmath.isinf(r):
                return cmath.nan

            return root(expr, r)

        if expr in [0, 1]:
            return expr

        if cmath.isinf(expr) or cmath.isnan(expr):
            return expr
     
        return root(expr, r)



class binomial(BinaryFunction):
    name = "binomial"

    def __init__(self, expr, r):
        super(binomial, self).__init__(expr, r)

    def latex(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class factorial(UnaryFunction):
    name = "!"

    def __init__(self, arg):
        super(factorial, self).__init__(arg)

    def latex(self):
        ss = self.arg.latex()
        if isinstance(self.arg, Function):
            ss = "(" + ss + ")"
        ss += "!"
        return ss

    def evaluate(self):
        raise NotImplementedError


class floor(UnaryFunction):
    name = "floor"

    def __init__(self, arg):
        super(floor, self).__init__(arg)

    def latex(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class conjugate(UnaryFunction):
    name = "conjugate"

    def __init__(self, arg):
        super(conjugate, self).__init__(arg)

    def latex(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class Sum(BinaryFunction):
    name = "Sum"

    def __init__(self, expr, r):
        super(Sum, self).__init__(expr, r)

    def latex(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class Product(BinaryFunction):
    name = "Product"

    def __init__(self, expr, r):
        super(Product, self).__init__(expr, r)

    def latex(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class Limit(Function):
    num_args = 4
    name = "Limit"

    def __init__(self, content, var, approaching, direction):
        super(Limit, self).__init__()
        self.content = objectify_input(content)
        self.var = objectify_input(var)
        self.approaching = objectify_input(approaching)
        self.direction = direction

    def latex(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def __str__(self):
        return self.name + "( " \
                    + self.content.__str__() + ", " \
                    + self.var.__str__() + ", "  \
                    + self.approaching.__str__ + ", " \
                    + self.direction.__str__ + " )"


class Negative(UnaryFunction):
    name = "Negative"
    def __init__(self, arg):
        super(Negative, self).__init__(arg)

    def latex(self):
        return "-" + self.arg.latex()

    def evaluate(self, **kwargs):
        arg_v = self.arg.evaluate(**kwargs)
        if isinstance(arg_v, MathObj):
            return Negative(arg_v)

        return -1* arg_v

