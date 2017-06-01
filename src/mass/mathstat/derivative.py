"""
Some classes to represent mathematical functions and operations.

Young-Il Joe
February 2017
"""

import math
import six
import numpy as np


class Function(object):
    """Base class for classes representing a mathematical function.
    This class provides some basic algebraic operations between Functions.
    
    Subclasses are supposed to implement derivative and __call__ methods.
    Though the Function is not defined explicitly as an abstract base class.
    
    lshift and rshift operators with another Function means a function composition.
    lshift and rshift operators with a number means a function evaluations.
    
    Examples
    --------
    PowerFunction(2) << ExponentialFunction()  # represents exp(x)**2
    x = Identity()
    (1 / (1 + x)) << ExponentialFunction() << -x  # represents a sigmoid function.
    1 / (1 + (ExponentialFunction() << -x))  # It's also a sigmoid function.
    2.0 >> PowerFunction(2)  # Its value is 4.0.
    PowerFunction(2) << 2.0  # Its value is also 4.0.
    """

    def __neg__(self):
        return -1 * self

    def __lshift__(self, other):
        if isinstance(other, (int, float)):
            return self(other)
        return Composition(self, other)

    def __rshift__(self, other):
        return other << self

    def __rrshift__(self, other):
        if isinstance(other, (int, float)):
            return self(other)
        return NotImplemented(">> is not implemented.")

    def __add__(self, other):
        return Summation(self, other)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return ConstantFunction(other) + self
        raise NotImplemented("+ is not implemented.")

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return ConstantFunction(other) - self
        raise NotImplemented("- is not implemented.")

    def __mul__(self, other):
        return Multiplication(self, other)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return ConstantFunction(other) * self
        raise NotImplemented("* is not implemented.")

    def truediv(self, other):
        if isinstance(other, (int, float)):
            other = ConstantFunction(other)
        return self * (PowerFunction(-1) << other)

    __truediv__ = truediv
    __div__ = truediv

    def rtruediv(self, other):
        if isinstance(other, (int, float)):
            return ConstantFunction(other) / self
        raise NotImplemented("/ is not implemented")

    __rtruediv__ = rtruediv
    __rdiv__ = rtruediv

    def __pow__(self, y):
        return PowerFunction(y) << self


class ConstantFunction(Function):
    def __init__(self, v):
        self.v = v

    def __call__(self, x, der=0):
        if der == 0:
            return np.ones_like(x) * self.v
        return np.zeros_like(x)

    def derivative(self, der=1):
        if der == 0:
            return self
        return ConstantFunction(0)

    def __repr__(self):
        return str(self.v)


class PowerFunction(Function):
    def __init__(self, n):
        self.n = n

    def derivative(self, der=1):
        if der == 0:
            return self
        if der == 1 and self.n == 1:
            return ConstantFunction(1)
        return Multiplication(ConstantFunction(self.n), PowerFunction(self.n - 1)).derivative(der=der - 1)

    def __call__(self, x, der=0):
        if der == 0:
            if self.n >= 0:
                return np.power(x, self.n)
            else:
                return 1.0 / np.power(x, -self.n)

        return self.derivative(der=der)(x, der=0)

    def __repr__(self):
        return str("x") + "^" + str(self.n)


class Identity(PowerFunction):
    def __init__(self):
        super(Identity, self).__init__(1)

    def __repr__(self):
        return "x"


class LogFunction(Function):
    def derivative(self, der=1):
        if der == 0:
            return self
        return PowerFunction(-1).derivative(der=der - 1)

    def __call__(self, x, der=0):
        if der == 0:
            return np.log(np.clip(x, 1e-6, None))
        else:
            return PowerFunction(-1)(x, der=der - 1)
        # return math.factorial(der - 1) * np.power(-1, der + 1) / np.power(x, der)

    def __repr__(self):
        return "log(x)"


class ExponentialFunction(Function):
    def derivative(self, der=1):
        return self

    def __call__(self, x, der=0):
        return np.exp(x)

    def __repr__(self):
        return "exp(x)"


class ExprMeta(type):
    def __call__(cls, a, b, *args, **kwarg):
        if cls is Summation:
            if isinstance(a, ConstantFunction) and isinstance(b, ConstantFunction):
                return ConstantFunction(a.v + b.v)
            if isinstance(a, ConstantFunction) and a.v == 0:
                return b
            if isinstance(b, ConstantFunction) and b.v == 0:
                return a
        elif cls is Composition:
            if isinstance(a, ConstantFunction):
                return a
            elif isinstance(a, Identity):
                return b
            elif isinstance(a, Multiplication) and isinstance(a.g, ConstantFunction):
                return Multiplication(a.g, Composition(a.h, b))
            elif isinstance(b, ConstantFunction):
                return ConstantFunction(a(b.v))
            elif isinstance(b, Identity):
                return a
            elif isinstance(a, PowerFunction) and isinstance(b, PowerFunction):
                return PowerFunction(a.n * b.n)
            elif isinstance(a, ExponentialFunction) and isinstance(b, LogFunction):
                return Identity()
            elif isinstance(a, LogFunction) and isinstance(b, ExponentialFunction):
                return Identity()
        elif cls is Multiplication:
            if isinstance(a, ConstantFunction) and isinstance(b, ConstantFunction):
                return ConstantFunction(a.v * b.v)

            # Because of the following statement,
            # if a is not a Constant function, then b is not a Constant function either.
            if isinstance(b, ConstantFunction):
                a, b = b, a

            if isinstance(a, ConstantFunction):
                if a.v == 0:
                    return a
                elif a.v == 1:
                    return b
                else:
                    if isinstance(b, Multiplication) and isinstance(b.g, ConstantFunction):
                        return Multiplication(ConstantFunction(a.v * b.g.v), b.h)
            if isinstance(a, PowerFunction):
                if isinstance(b, PowerFunction):
                    return PowerFunction(a.n + b.n)
                elif isinstance(b, Multiplication) and isinstance(b.h, PowerFunction):
                    return Multiplication(b.g, PowerFunction(a.n + b.h.n))

            if isinstance(a, Multiplication) and isinstance(a.g, ConstantFunction):
                if isinstance(b, Multiplication) and isinstance(b.g, ConstantFunction):
                    return Multiplication(ConstantFunction(a.g.v * b.g.v), Multiplication(a.h, b.h))

            if isinstance(a, Multiplication) and isinstance(a.h, PowerFunction):
                if isinstance(b, PowerFunction):
                    return Multiplication(a.g, PowerFunction(a.h.n + b.n))
                elif isinstance(b, Multiplication) and isinstance(b.h, PowerFunction):
                    return Multiplication(Multiplication(a.g, b.g), PowerFunction(a.h.n + b.h.n))

        return type.__call__(cls, a, b, *args, **kwarg)


class BinaryOperation(six.with_metaclass(ExprMeta)):
    def __init__(self, g, h):
        super(BinaryOperation, self).__init__()
        self.g = g
        self.h = h


class Composition(BinaryOperation, Function):
    def derivative(self, der=1):
        if der == 0:
            return self
        m = Multiplication(Composition(self.g.derivative(der=1), self.h), self.h.derivative(der=1))
        return m.derivative(der=der - 1)

    def __call__(self, x, der=0):
        if der == 0:
            return self.g(self.h(x))

        return self.derivative(der=der)(x)

    def __repr__(self):
        if six.PY2:
            return ("(" + str(self.g) + u" \u2022 " + str(self.h) + ")").encode("utf8")
        elif six.PY3:
            return "(" + str(self.g) + " \u2022 " + str(self.h) + ")"


class Multiplication(BinaryOperation, Function):
    def derivative(self, der=1):
        if der == 0:
            return self

        return Summation(Multiplication(self.g.derivative(der=1), self.h),
                         Multiplication(self.g, self.h.derivative(der=1))).derivative(der=der - 1)

    def __call__(self, x, der=0):
        if der == 0:
            return self.g(x) * self.h(x)

        return np.sum([self.g(x, der=n) * self.h(x, der=der - n) *
                       math.factorial(der) / math.factorial(der - n) / math.factorial(n)
                       for n in range(der + 1)], axis=0)

    def __repr__(self):
        if isinstance(self.g, ConstantFunction) and self.g.v == -1:
            return "(-" + str(self.h) + ")"
        return "(" + str(self.g) + " * " + str(self.h) + ")"


class Summation(BinaryOperation, Function):
    def derivative(self, der=1):
        if der == 0:
            return self

        return Summation(self.g.derivative(der=der), self.h.derivative(der=der))

    def __call__(self, x, der=0):
        return self.g(x, der=der) + self.h(x, der=der)

    def __repr__(self):
        return "(" + str(self.g) + " + " + str(self.h) + ")"
