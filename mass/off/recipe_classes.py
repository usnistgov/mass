import mass
from dataclasses import dataclass
import numpy as np

@dataclass
class TruesOfSameSize:
    def __call__(self, x):
        return np.ones(len(x), dtype="bool")
    
@dataclass
class Subtract:
    right: float
    def __call__(self, left):
        return left-self.right
    
@dataclass
class Add:
    right: float
    def __call__(self, left):
        return left+self.right
    
@dataclass 
class MatMulAB_FixedB:
    B: any
    def __call__(self, A):
        return np.matmul(A,self.B)
    
@dataclass 
class MatMulAB_FixedA:
    A: any
    def __call__(self, B):
        return np.matmul(self.A,B)
    
@dataclass
class SubtractThenScale:
    sub: float
    scale: float
    def __call__(self, x):
        return self.scale*(x-self.sub)
    
@dataclass
class DivideTwo:
    def __call__(self, x, y):
        return x/y

@dataclass
class ScalarMultAndTurnToInt64:
    mult: float
    def __call__(self, x):
        return np.array(x, dtype=np.int64)*self.mult