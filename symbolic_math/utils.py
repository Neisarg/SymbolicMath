import symbolic_math as sm

def is_number(num):
    if type(num) in [int, float]:
        return True
    else:
        return False

def is_string(c):
    if type(c) == str:
        return True
    else:
        return False

