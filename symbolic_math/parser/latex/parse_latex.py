import antlr4
from antlr4.error.ErrorListener import ErrorListener

try:
    from .gen.PSParser import PSParser
    from .gen.PSLexer import PSLexer
    from .gen.PSListener import PSListener
except:
    from gen.PSParser import PSParser
    from gen.PSLexer import PSLexer
    from gen.PSListener import PSListener

import symbolic_math.functions as smb
import symbolic_math as sm

from symbolic_math.manipulation import subsitute


def parse_latex(expression):
    matherror = MathErrorListener(expression)

    stream = antlr4.InputStream(expression)
    lex = PSLexer(stream)
    lex.removeErrorListeners()
    lex.addErrorListener(matherror)

    tokens = antlr4.CommonTokenStream(lex)
    parser = PSParser(tokens)

    # remove default console error listener
    parser.removeErrorListeners()
    parser.addErrorListener(matherror)

    relation = parser.math().relation()
    expr = convert_relation(relation)

    return expr


class MathErrorListener(ErrorListener):
    def __init__(self, src):
        super(ErrorListener, self).__init__()
        self.src = src

    def syntaxError(self, recog, symbol, line, col, msg, e):
        fmt = "%s\n%s\n%s"
        marker = "~" * col + "^"

        if msg.startswith("missing"):
            err = fmt % (msg, self.src, marker)
        elif msg.startswith("no viable"):
            err = fmt % ("I expected something else here", self.src, marker)
        elif msg.startswith("mismatched"):
            names = PSParser.literalNames
            expected = [names[i] for i in e.getExpectedTokens() if i < len(names)]
            if expected < 10:
                expected = " ".join(expected)
                err = (fmt % ("I expected one of these: " + expected,
                              self.src, marker))
            else:
                err = (fmt % ("I expected something else here", self.src, marker))
        else:
            err = fmt % ("I don't understand this", self.src, marker)
        raise Exception(err)


def convert_relation(rel):
    if rel.expr():
        return convert_expr(rel.expr())

    lh = convert_relation(rel.relation(0))
    rh = convert_relation(rel.relation(1))
    if rel.LT():
        return smb.LT(lh, rh)
    elif rel.LTE():
        return smb.LEQ(lh, rh)
    elif rel.GT():
        return smb.GT(lh, rh)
    elif rel.GTE():
        return smb.GEQ(lh, rh)
    elif rel.EQUAL():
        return smb.EQ(lh, rh)


def convert_expr(expr):
    return convert_add(expr.additive())


def convert_add(add):
    if add.ADD():
       lh = convert_add(add.additive(0))
       rh = convert_add(add.additive(1))
       return smb.Add(lh, rh)
    elif add.SUB():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        return smb.Subtract(lh, rh)
    else:
        return convert_mp(add.mp())


def convert_mp(mp):
    if hasattr(mp, 'mp'):
        mp_left = mp.mp(0)
        mp_right = mp.mp(1)
    else:
        mp_left = mp.mp_nofunc(0)
        mp_right = mp.mp_nofunc(1)

    if mp.MUL() or mp.CMD_TIMES() or mp.CMD_CDOT():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        return smb.Multiply(lh, rh)
    elif mp.DIV() or mp.CMD_DIV() or mp.COLON():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        return smb.Divide(lh, rh)
    else:
        if hasattr(mp, 'unary'):
            return convert_unary(mp.unary())
        else:
            return convert_unary(mp.unary_nofunc())


def convert_unary(unary):
    if hasattr(unary, 'unary'):
        nested_unary = unary.unary()
    else:
        nested_unary = unary.unary_nofunc()
    if hasattr(unary, 'postfix_nofunc'):
        first = unary.postfix()
        tail = unary.postfix_nofunc()
        postfix = [first] + tail
    else:
        postfix = unary.postfix()

    if unary.ADD():
        return convert_unary(nested_unary)
    elif unary.SUB():
        numabs = convert_unary(nested_unary)
        # Use Integer(-n) instead of Mul(-1, n)
        return smb.Negative(numabs)
    elif postfix:
        return convert_postfix_list(postfix)


def convert_postfix_list(arr, i=0):
    if i >= len(arr):
        raise Exception("Index out of bounds")

    res = convert_postfix(arr[i])
    if isinstance(res, smb.Function):
        if i == len(arr) - 1:
            return res # nothing to multiply by
        else:
            if i > 0:
                left = convert_postfix(arr[i - 1])
                right = convert_postfix(arr[i + 1])
                if isinstance(left, smb.Function) and isinstance(right, smb.Function):
                    left_syms = convert_postfix(arr[i - 1]).atoms(smb.symbol_constructor)
                    right_syms = convert_postfix(arr[i + 1]).atoms(smb.symbol_constructor)
                    # if the left and right sides contain no variables and the
                    # symbol in between is 'x', treat as multiplication.
                    if len(left_syms) == 0 and len(right_syms) == 0 and str(res) == "x":
                        return convert_postfix_list(arr, i + 1)
            # multiply by next
            return smb.Multiply(res, convert_postfix_list(arr, i + 1))
    elif isinstance(res, smb.Symbol):
        return res
    else: # must be derivative
        wrt = res[0]
        if i == len(arr) - 1:
            raise Exception("Expected expression for derivative")
        else:
            expr = convert_postfix_list(arr, i + 1)
            return smb.Derivative(expr, wrt)

def do_subs(expr, at):
    if at.expr():
        at_expr = convert_expr(at.expr())
        syms = at_expr.atoms(smb.symbol_constructor)
        if len(syms) == 0:
            return expr
        elif len(syms) > 0:
            sym = next(iter(syms))
            return subsitute(expr, sym, at_expr)
    elif at.equality():
        lh = convert_expr(at.equality().expr(0))
        rh = convert_expr(at.equality().expr(1))
        return subsitute(expr, lh, rh)


def convert_postfix(postfix):
    if hasattr(postfix, 'exp'):
        exp_nested = postfix.exp()
    else:
        exp_nested = postfix.exp_nofunc()

    exp = convert_exp(exp_nested)

    for op in postfix.postfix_op():
        if op.BANG():
            if isinstance(exp, list):
                raise Exception("Cannot apply postfix to derivative")
            raise Exception("Factorial not implemented")
            exp = smb.factorial(exp)
        elif op.eval_at():
            ev = op.eval_at()
            at_b = None
            at_a = None
            if ev.eval_at_sup():
                at_b = do_subs(exp, ev.eval_at_sup())
            if ev.eval_at_sub():
                at_a = do_subs(exp, ev.eval_at_sub())
            if at_b != None and at_a != None:
                exp = smb.Subtract(at_b, at_a)
            elif at_b != None:
                exp = at_b
            elif at_a != None:
                exp = at_a
    return exp

def convert_exp(exp):
    if hasattr(exp, 'exp'):
        exp_nested = exp.exp()
    else:
        exp_nested = exp.exp_nofunc()

    if exp_nested:
        base = convert_exp(exp_nested)
        if isinstance(base, list):
            raise Exception("Cannot raise derivative to power")
        if exp.atom():
            exponent = convert_atom(exp.atom())
        elif exp.expr():
            exponent = convert_expr(exp.expr())
        return smb.Pow(base, exponent)
    else:
        if hasattr(exp, 'comp'):
            return convert_comp(exp.comp())
        else:
            return convert_comp(exp.comp_nofunc())



def convert_comp(comp):
    if comp.group():
        return convert_expr(comp.group().expr())
    elif comp.abs_group():
        return smb.Abs(convert_expr(comp.abs_group().expr()))
    elif comp.atom():
        return convert_atom(comp.atom())
    elif comp.frac():
        return convert_frac(comp.frac())
    # elif comp.binom():
    #     return convert_binom(comp.binom())
    # elif comp.floor():
    #     return convert_floor(comp.floor())
    # elif comp.ceil():
    #     return convert_ceil(comp.ceil())
    elif comp.func():
        return convert_func(comp.func())


def convert_atom(atom):
    if atom.LETTER():
        subscriptName = ''
        if atom.subexpr():
            subscript = None
            if atom.subexpr().expr():  # subscript is expr
                subscript = convert_expr(atom.subexpr().expr())
            else:  # subscript is atom
                subscript = convert_atom(atom.subexpr().atom())
            subscriptName = '_{' + subscript.__str__() + '}'
        return smb.symbol_constructor(atom.LETTER().getText() + subscriptName)
    elif atom.SYMBOL():
        s = atom.SYMBOL().getText()[1:]
        if s == "infty":
            return smb.infty
        else:
            if atom.subexpr():
                subscript = None
                if atom.subexpr().expr():  # subscript is expr
                    subscript = convert_expr(atom.subexpr().expr())
                else:  # subscript is atom
                    subscript = convert_atom(atom.subexpr().atom())
                #subscriptName = StrPrinter().doprint(subscript)
                subscriptName = subscript.__str__()
                s += '_{' + subscriptName + '}'
            return smb.symbol_constructor(s)
    elif atom.NUMBER():
        s = atom.NUMBER().getText().replace(",", "")
        return smb.Number(s)
    elif atom.DIFFERENTIAL():
        var = get_differential_var(atom.DIFFERENTIAL())
        return smb.symbol_constructor('d' + var.name)
    elif atom.mathit():
        text = rule2text(atom.mathit().mathit_text())
        return smb.symbol_constructor(text)
    elif atom.bra():
        val = convert_expr(atom.bra().expr())
        return Bra(val)
    elif atom.ket():
        val = convert_expr(atom.ket().expr())
        return Ket(val)


def rule2text(ctx):
    stream = ctx.start.getInputStream()
    # starting index of starting token
    startIdx = ctx.start.start
    # stopping index of stopping token
    stopIdx = ctx.stop.stop

    return stream.getText(startIdx, stopIdx)


def convert_frac(frac):
    diff_op = False
    partial_op = False
    lower_itv = frac.lower.getSourceInterval()
    lower_itv_len = lower_itv[1] - lower_itv[0] + 1
    if (frac.lower.start == frac.lower.stop and
            frac.lower.start.type == PSLexer.DIFFERENTIAL):
        wrt = get_differential_var_str(frac.lower.start.text)
        diff_op = True
    elif (lower_itv_len == 2 and
          frac.lower.start.type == PSLexer.SYMBOL and
          frac.lower.start.text == '\\partial' and
          (frac.lower.stop.type == PSLexer.LETTER or frac.lower.stop.type == PSLexer.SYMBOL)):
        partial_op = True
        wrt = frac.lower.stop.text
        if frac.lower.stop.type == PSLexer.SYMBOL:
            wrt = wrt[1:]

    if diff_op or partial_op:
        wrt = smb.symbol_constructor(wrt)
        if (diff_op and frac.upper.start == frac.upper.stop and
                frac.upper.start.type == PSLexer.LETTER and
                frac.upper.start.text == 'd'):
            return [wrt]
        elif (partial_op and frac.upper.start == frac.upper.stop and
              frac.upper.start.type == PSLexer.SYMBOL and
              frac.upper.start.text == '\\partial'):
            return [wrt]
        upper_text = rule2text(frac.upper)

        expr_top = None
        if diff_op and upper_text.startswith('d'):
            expr_top = process_latex(upper_text[1:])
        elif partial_op and frac.upper.start.text == '\\partial':
            expr_top = process_latex(upper_text[len('\\partial'):])
        if expr_top:
            return smb.Derivative(expr_top, wrt)

    expr_top = convert_expr(frac.upper)
    expr_bot = convert_expr(frac.lower)
    inverse_denom = smb.Pow(expr_bot, -1)
    if expr_top == 1:
        return inverse_denom
    else:
        return smb.Divide(expr_top, expr_bot)

def convert_binom(binom):
    expr_n = convert_expr(binom.n)
    expr_k = convert_expr(binom.k)
    return smb.binomial(expr_n, expr_k)

def convert_floor(floor):
    val = convert_expr(floor.val)
    return smb.floor(val)

def convert_ceil(ceil):
    val = convert_expr(ceil.val)
    return smb.ceiling(val)

def convert_func(func):
    if func.func_normal():
        if func.L_PAREN():  # function called with parenthesis
            arg = convert_func_arg(func.func_arg())
        else:
            arg = convert_func_arg(func.func_arg_noparens())

        name = func.func_normal().start.text[1:]

        # change arc<trig> -> a<trig>
        if name in ["arcsin", "arccos", "arctan", "arccsc", "arcsec",
                    "arccot"]:
            name = "a" + name[3:]
            expr = getattr(sm.trigonometry, name)(arg)
        if name in ["arsinh", "arcosh", "artanh"]:
            name = "a" + name[2:]
            expr = getattr(sm.trigonometry, name)(arg)

        if name == "exp":
            expr = smb.exp(arg)

        if (name == "log" or name == "ln"):
            if func.subexpr():
                base = convert_expr(func.subexpr().expr())
                expr = smb.log_b(arg, base)
            elif name == "log":
                expr = smb.log(arg)
            elif name == "ln":
                expr = smb.ln(arg)
            

        func_pow = None
        should_pow = True
        if func.supexpr():
            if func.supexpr().expr():
                func_pow = convert_expr(func.supexpr().expr())
            else:
                func_pow = convert_atom(func.supexpr().atom())

        if name in ["sin", "cos", "tan", "csc", "sec", "cot", "sinh", "cosh", "tanh"]:
            if func_pow == -1:
                name = "a" + name
                should_pow = False
            expr = getattr(sm.trigonometry, name)(arg)

        if func_pow and should_pow:
            expr = smb.Pow(expr, func_pow)

        return expr
    elif func.LETTER() or func.SYMBOL():
        if func.LETTER():
            fname = func.LETTER().getText()
        elif func.SYMBOL():
            fname = func.SYMBOL().getText()[1:]
        fname = str(fname)  # can't be unicode
        if func.subexpr():
            subscript = None
            if func.subexpr().expr():  # subscript is expr
                subscript = convert_expr(func.subexpr().expr())
            else:  # subscript is atom
                subscript = convert_atom(func.subexpr().atom())
            #subscriptName = StrPrinter().doprint(subscript)
            subscriptName = subscript.__str__()
            fname += '_{' + subscriptName + '}'
        input_args = func.args()
        output_args = []
        while input_args.args():  # handle multiple arguments to function
            output_args.append(convert_expr(input_args.expr()))
            input_args = input_args.args()
        output_args.append(convert_expr(input_args.expr()))
        return smb.GenericFunction(fname, *output_args)
    elif func.FUNC_INT():
        return handle_integral(func)
    elif func.FUNC_SQRT():
        expr = convert_expr(func.base)
        if func.root:
            r = convert_expr(func.root)
            return smb.root(expr, r)
        else:
            return smb.sqrt(expr)
    elif func.FUNC_OVERLINE():
        expr = convert_expr(func.base)
        return smb.conjugate(expr)
    elif func.FUNC_SUM():
        return handle_sum_or_prod(func, "summation")
    elif func.FUNC_PROD():
        return handle_sum_or_prod(func, "product")
    elif func.FUNC_LIM():
        return handle_limit(func)


def convert_func_arg(arg):
    if hasattr(arg, 'expr'):
        return convert_expr(arg.expr())
    else:
        return convert_mp(arg.mp_nofunc())


def handle_integral(func):
    if func.additive():
        integrand = convert_add(func.additive())
    elif func.frac():
        integrand = convert_frac(func.frac())
    else:
        integrand = 1

    int_var = None
    if func.DIFFERENTIAL():
        int_var = get_differential_var(func.DIFFERENTIAL())
    else:
        for sym in integrand.atoms(smb.symbol_constructor):
            s = str(sym)
            if len(s) > 1 and s[0] == 'd':
                if s[1] == '\\':
                    int_var = smb.symbol_constructor(s[2:])
                else:
                    int_var = smb.symbol_constructor(s[1:])
                int_sym = sym
        if int_var:
            integrand = integrand.subs(int_sym, 1)
        else:
            # Assume dx by default
            int_var = smb.symbol_constructor('x')

    if func.subexpr():
        if func.subexpr().atom():
            lower = convert_atom(func.subexpr().atom())
        else:
            lower = convert_expr(func.subexpr().expr())
        if func.supexpr().atom():
            upper = convert_atom(func.supexpr().atom())
        else:
            upper = convert_expr(func.supexpr().expr())
        return smb.Integral(integrand, (int_var, lower, upper))
    else:
        return smb.Integral(integrand, int_var)


def handle_sum_or_prod(func, name):
    val = convert_mp(func.mp())
    iter_var = convert_expr(func.subeq().equality().expr(0))
    start = convert_expr(func.subeq().equality().expr(1))
    if func.supexpr().expr():  # ^{expr}
        end = convert_expr(func.supexpr().expr())
    else:  # ^atom
        end = convert_atom(func.supexpr().atom())

    if name == "summation":
        return smb.Sum(val, (iter_var, start, end))
    elif name == "product":
        return smb.Product(val, (iter_var, start, end))


def handle_limit(func):
    sub = func.limit_sub()
    if sub.LETTER():
        var = smb.symbol_constructor(sub.LETTER().getText())
    elif sub.SYMBOL():
        var = smb.symbol_constructor(sub.SYMBOL().getText()[1:])
    else:
        var = smb.symbol_constructor('x')
    if sub.SUB():
        direction = "-"
    else:
        direction = "+"
    approaching = convert_expr(sub.expr())
    content = convert_mp(func.mp())

    return smb.Limit(content, var, approaching, direction)


def get_differential_var(d):
    text = get_differential_var_str(d.getText())
    return smb.symbol_constructor(text)


def get_differential_var_str(text):
    for i in range(1, len(text)):
        c = text[i]
        if not (c == " " or c == "\r" or c == "\n" or c == "\t"):
            idx = i
            break
    text = text[idx:]
    if text[0] == "\\":
        text = text[1:]
    return text


def test_parsing():
    #eq = "a^3 + b^3 + c^3 - 3*a*b*c = ( a + b + c ) * ( a^2 + b^2 + c^2 - a*b - b*c - c*a )"
    #eq = "a^{3}+b^{3}+c^{3}-\\left(\\left(3a\\right)b\\right)c=\\left(a+b+c\\right)\\left(a^{2}+b^{2}+c^{2}-a-b-c\\right)"
    eq = "\\alpha*(222+\\beta)"
    #eq = "a^3 + b^3 + c^3 - 3*a*b*c"
    print(eq)
    ll = parse_latex(eq)
    rev = ll.latex()
    print(eq, ll, rev)
    print(type(ll), type(ll.left), type(ll.right))

if __name__ == "__main__":
    test_parsing()
