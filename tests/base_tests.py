from symbolic_math.base import *
#from symbolic_math.functions import *
from symbolic_math import parse_latex
import symbolic_math as sm

def test_subsitute():
    x = sm.Symbol('x')
    eq = sm.Add(sm.Add(x, "alpha"), "gamma")
    print(eq.latex())
    eq = sm.subsitute(eq, "alpha", 1)
    print(eq)
    eq = sm.subsitute(eq, "Add", "Subtract")
    print(eq)

def test_add_evaluate():
    eq = "\\gamma*\\frac{\\alpha}{\\beta}"
    eq = parse_latex(eq)
    print(eq, type(eq))
    val = eq.evaluate(alpha = 0, beta=1)
    print(eq, val)

def test_get_func():
    klass = sm.get_function_by_name("Add")
    print(klass)

if __name__ == "__main__":
    #test_add()
    #test_add_evaluate()    
    #test_get_func()
    test_subsitute()