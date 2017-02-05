import math
import six

import numpy as np
from scipy.interpolate import splev


class ConstantFunction(object):
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


# class CubicSplineFunction:
#     def __init__(self, knots, coeffs, der=0):
#         self.knots = knots
#         self.coeffs = coeffs
#         self.der = der
#
#     def derivative(self, der=1):
#         if self.der + der > 3:
#             return ConstantFunction(0)
#         return CubicSplineFunction(self.knots, self.coeffs, der=self.der + der)
#
#     def __call__(self, x, der=0):
#         if self.der + der > 3:
#             return np.zeros_like(x)
#
#         return splev(x, (self.knots, self.coeffs, 3), der=self.der + der)
#
#     def __repr__(self):
#         return "CubicSpline" + ("".join(["\""] * self.der)) + "(x)"


class PowerFunction(object):
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
            return np.power(x, self.n)

        return self.derivative(der=der)(x, der=0)

    def __repr__(self):
        return str("x") + "^" + str(self.n)


class Identity(PowerFunction):
    def __init__(self):
        super(Identity, self).__init__(1)

    def __repr__(self):
        return "x"


class LogFunction(object):
    def derivative(self, der=1):
        if der == 0:
            return self
        return PowerFunction(-1).derivative(der=der - 1)

    def __call__(self, x, der=0):
        if der == 0:
            return np.log(x)
        else:
            return PowerFunction(-1)(x, der=der - 1)
        # return math.factorial(der - 1) * np.power(-1, der + 1) / np.power(x, der)

    def __repr__(self):
        return "log(x)"


class ExponentialFunction(object):
    def derivative(self, der=1):
        return self

    def __call__(self, x, der=0):
        return np.exp(x)

    def __repr__(self):
        return "exp(x)"


class ExprMeta(type):
    def __call__(cls, *args, **kwarg):
        if cls is Summation:
            a, b = args[:2]
            if isinstance(a, ConstantFunction) and isinstance(b, ConstantFunction):
                return ConstantFunction(a.v + b.v)
            if isinstance(a, ConstantFunction) and a.v == 0:
                return b
            if isinstance(b, ConstantFunction) and b.v == 0:
                return a
        elif cls is Composition:
            a, b = args[:2]
            if isinstance(a, ConstantFunction):
                return a
            elif isinstance(b, ConstantFunction):
                return ConstantFunction(a(b.v))
        elif cls is Multiplication:
            a, b = args[:2]
            if isinstance(a, ConstantFunction) and isinstance(b, ConstantFunction):
                return ConstantFunction(a.v * b.v)

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

        return type.__call__(cls, *args, **kwarg)


class BinaryOperation(six.with_metaclass(ExprMeta)):
    def __init__(self, g, h):
        self.g = g
        self.h = h


class Composition(BinaryOperation):
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


class Multiplication(BinaryOperation):
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
        return "(" + str(self.g) + " * " + str(self.h) + ")"


class Summation(BinaryOperation):
    def derivative(self, der=1):
        if der == 0:
            return self

        return Summation(self.g.derivative(der=der), self.h.derivative(der=der))

    def __call__(self, x, der=0):
        return self.g(x, der=der) + self.h(x, der=der)

    def __repr__(self):
        return "(" + str(self.g) + " + " + str(self.h) + ")"
