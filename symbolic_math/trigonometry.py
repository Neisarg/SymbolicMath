from symbolic_math.base import *
from symbolic_math.utils import *
import math
import cmath

class asin(UnaryFunction):
    name = "asin"

    def __init__(self, arg):
        super(asin, self).__init__(arg)

    def latex(self):
        return "\\operatorname{arcsin} (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return asin(av)

        v = cmath.asin(av)
        if v.imag == 0:
            return v.real
        else:
            return v


class acos(UnaryFunction):
    name = "acos"

    def __init__(self, arg):
        super(acos, self).__init__(arg)

    def latex(self):
        return "\\operatorname{arccos} (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return acos(av)
            
        v = cmath.acos(av)
        if v.imag == 0:
            return v.real
        else:
            return v


class atan(UnaryFunction):
    name = "atan"

    def __init__(self, arg):
        super(atan, self).__init__(arg)

    def latex(self):
        return "\\operatorname{arctan} (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return atan(av)
            
        v = cmath.atan(av)
        if v.imag == 0:
            return v.real
        else:
            return v


class acsc(UnaryFunction):
    name = "acsc"

    def __init__(self, arg):
        super(acsc, self).__init__(arg)

    def latex(self):
        return "\\operatorname{arccsc} (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return acsc(av)
        
        if av == 0:
            return cmath.asin(cmath.inf)   
        v = cmath.asin(1.0/av)
        if v.imag == 0:
            return v.real
        else:
            return v


class asec(UnaryFunction):
    name = "asec"

    def __init__(self, arg):
        super(asec, self).__init__(arg)

    def latex(self):
        return "\\operatorname{arcsec} (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return asec(av)
        
        if av == 0:
            return cmath.acos(cmath.inf) 

        v = cmath.acos(1.0/av)
        if v.imag == 0:
            return v.real
        else:
            return v


class acot(UnaryFunction):
    name = "acot"
    
    def __init__(self, arg):
        super(acot, self).__init__(arg)

    def latex(self):
        return "\\operatorname{arccot} (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return acot(av)
         
        if av == 0:
            return cmath.atan(cmath.inf) 

        v = cmath.atan(1.0/av)
        if v.imag == 0:
            return v.real
        else:
            return v


class asinh(UnaryFunction):
    name = "asinh"
    
    def __init__(self, arg):
        super(asinh, self).__init__(arg)

    def latex(self):
        return "\\operatorname{arsinh} (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return asinh(av)
            
        v = cmath.asinh(av)
        if v.imag == 0:
            return v.real
        else:
            return v


class acosh(UnaryFunction):
    name = "acosh"
    
    def __init__(self, arg):
        super(acosh, self).__init__(arg)

    def latex(self):
        return "\\operatorname{arcosh} (" + self.arg.latex() + ")"

    def evaluate(self,  **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return acosh(av)
            
        v = cmath.acosh(av)
        if v.imag == 0:
            return v.real
        else:
            return v

class atanh(UnaryFunction):
    name = "atanh"
    
    def __init__(self, arg):
        super(atanh, self).__init__(arg)

    def latex(self):
        return "\\operatorname{artanh} (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return atanh(av)
            
        v = cmath.atanh(av)
        if v.imag == 0:
            return v.real
        else:
            return v


class sin(UnaryFunction):
    name = "sin"

    def __init__(self, arg):
        super(sin, self).__init__(arg)

    def latex(self):
        return "\\sin (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return sin(av)
            
        v = cmath.sin(av)
        if v.imag == 0:
            return v.real
        else:
            return v


class cos(UnaryFunction):
    name = "cos"

    def __init__(self, arg):
        super(cos, self).__init__(arg)

    def latex(self):
        return "\\cos (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return cos(av)
            
        v = cmath.cos(av)
        if v.imag == 0:
            return v.real
        else:
            return v


class tan(UnaryFunction):
    name = "tan"

    def __init__(self, arg):
        super(tan, self).__init__(arg)

    def latex(self):
        return "\\tan (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return tan(av)
            
        v = cmath.atan(av)
        if v.imag == 0:
            return v.real
        else:
            return v


class csc(UnaryFunction):
    name = "csc"

    def __init__(self, arg):
        super(csc, self).__init__(arg)

    def latex(self):
        return "\\csc (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return csc(av)
        
        vv = cmath.sin(av)
        if vv == 0:
            return cmath.inf
        v = 1.0/vv
        if v.imag == 0:
            return v.real
        else:
            return v


class sec(UnaryFunction):
    name = "sec"

    def __init__(self, arg):
        super(sec, self).__init__(arg)

    def latex(self):
        return "\\sec (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return sec(av)
        
        vv = cmath.cos(av)
        if vv == 0:
            return cmath.inf
        v = 1.0/vv

        if v.imag == 0:
            return v.real
        else:
            return v


class cot(UnaryFunction):
    name = "cot"
    
    def __init__(self, arg):
        super(cot, self).__init__(arg)

    def latex(self):
        return "\\cot (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return cot(av)
        
        vv = cmath.tan(av)
        if vv == 0:
            return cmath.inf
        v = 1.0/vv
    
        if v.imag == 0:
            return v.real
        else:
            return v


class sinh(UnaryFunction):
    name = "sinh"
    
    def __init__(self, arg):
        super(sinh, self).__init__(arg)

    def latex(self):
        return "\\operatorname{sinh} (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return sinh(av)
            
        v = cmath.sinh(av)
        if v.imag == 0:
            return v.real
        else:
            return v


class cosh(UnaryFunction):
    name = "cosh"
    
    def __init__(self, arg):
        super(cosh, self).__init__(arg)

    def latex(self):
        return "\\operatorname{cosh} (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return cosh(av)
            
        v = cmath.cosh(av)
        if v.imag == 0:
            return v.real
        else:
            return v

class tanh(UnaryFunction):
    name = "tanh"
    
    def __init__(self, arg):
        super(tanh, self).__init__(arg)

    def latex(self):
        return "\\operatorname{tanh} (" + self.arg.latex() + ")"

    def evaluate(self, **kwargs):
        av = self.arg.evaluate(**kwargs)
        if isinstance(av, MathObj):
            return tanh(av)
            
        v = cmath.tanh(av)
        if v.imag == 0:
            return v.real
        else:
            return v