import enum
from re import S
from typing import Type
from mpmath.functions.functions import re
import numpy as np
# import tensorflow as tf
import sympy
from sympy import Derivative

default_metric = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

_mutable_types = [
    int, float, 
    np.float32, np.int32,
    np.float16, np.int16,
]
_mutable_objs = []

def make_mutable(obj):
    """Flag this object as mutable, mutable objects will be 
    moved to the front of products."""
    _mutable_objs.append(obj)
def make_mutable_type(type_):
    """Flag objects of this type and subtypes as mutable, objects of this type
    will be moved to the front of products."""
    _mutable_types.append(type_)
def is_mutable(obj):
    if obj in _mutable_objs:
        return True
    for t in _mutable_types:
        if isinstance(obj, t):
            return True
    return False
    
def int_like(data):
    data = np.array(data)
    try:
        data / 2
    except Exception:
        return False
    else:
        return (data // 1 == data)

def is_scalar(data):
    try: 
        data/2
    except:
        return False
    else:
        return len(np.shape(data)) == 0

def is_num(obj, **indices):
    try:
        return obj.is_num(**indices)
    except:
        return is_scalar(obj)

def list_swap(data, key, new):
    " Swap element with value 'key' with value 'new', performs in place "
    ret = list(data)
    if key in data:
        ret[ret.index(key)] = new
    if key[0] == '/':
        # Disregard this
        key = key[1:]
        # But leave the new one as is
    
    # try again:
    for i, val in enumerate(ret):
        if type(val) is str and val[0] == '/':
            val = val[1:]
        if val == key:
            ret[i] = new
            break
    return ret

def evaluate(obj, **indices):
    try:
        return obj.evaluate(**indices)
    except:
        return obj

def _get_met_indices(metric):
    """ Returns a list of indices unlikely to be used by the user
    to allow for raising/lowering of indices via the given metric. """
    letters = [f"/{x}" for x in range(0, len(metric.shape))]
    return letters

class ESTensor:
    """ Class for a Tensor with Einstein Summation functionality.
    Assumes that the given data is of the Contravariant (upper index) representation.
    
    Can optionally provide a metric to allow computation of lower indices, otherwise
    an identity metric will be used. The metric should be the Covariant metric.
    """
    metric = default_metric
    def __init__(self, name, data, metric=None):
        self.name = name
        try:
            data.shape
        except:
            data = np.array(data)
        finally:
            self.shape = data.shape
        self.data = data
        self.dim = len(self.shape)        
        if metric is not None:
            self.metric = metric


    def __call__(self, *indices):
        """ Creates an IndexedTensor representation of this tensor, 
        which can be multiplied by other indexed tensors. When multiplied,
        if two IndexedTensors have repeated indices, they will be summed over.

        Prefix any index with an underscore (eg. `_j`) to mark it as a lower index.
        """
        if len(indices) != self.dim:
            raise ValueError("This tensor is rank " f"{self.dim}, please provide this many indices.")


        indices = np.array(indices, dtype=object)
        if np.all(int_like(indices)):
            return RankOneTensor(self.data[tuple(indices)])
        else:
            met_indices = _get_met_indices(self.metric)
            met_muls = []
            for i, index in enumerate(indices):
                # Apply metric transformation
                if type(index) is str and index[0] == '_':
                    met_muls.append(
                        ESTensor("g", self.metric)(
                            f"/{index[1:]}", *met_indices[1:]
                        )
                    )
                    indices = list_swap(indices, index, met_indices[len(met_muls)])
            ret = IndexedTensor(self, *indices)
            for p in met_muls:
                ret = p * ret

            return ret

    def __mul__(self, other):
        try:
            return ESTensor("*" + self.name, self.data * other, self.metric)
        except Exception as e:
            raise e

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        return self.name



class _CompositeTensors:
    _blocking = False # Don't let terms to the right of this one be brought forward
    def evaluate(self): pass
    def __mul__(self, other): pass
    def __add__(self, other): pass
    def __abs__(self): return self # TODO
    def contract(self): pass
    def expand(self): return self
    def permute(self): return self
    def fully_qualified(self): return False
    def __repr__(self) -> str:
        return self.__str__()
    @property
    def num_children(self):
        return len(self.children)


# class DifferentialOperator:
#     """ Acts as a differential operator which contains an expression A, 
#     and when multiplied by an expression B on the right, returns the derivative
#     of B with respect to A, ie dB/dA.
#     """
#     def __init__(self, expression):
#         """ Create a differential operator. Provide the expression for differentiation,
#         ie for d/dx provide the symbol x.
#         """
#         self.var = expression
    
#     def __mul__(self, other):
#         try:
#             return sympy.Derivative(other, self.var)
#         except TypeError:
#             # I dunno what I was gonna do here
#             pass

#     # def __rmul__(self, other):
#     #     return DifferentialOperator()

class RankOneTensor(_CompositeTensors):
    def __init__(self, scalar):
        if not is_scalar(scalar):
            print("Got", scalar)
        if type(scalar) == RankOneTensor:
            scalar = scalar.evaluate()
        self.value = scalar

    def evaluate(self, **indices):
        return self.value
    
    def is_num(self, *args, **kwargs):
        return True

    def __add__(self, other):
        return RankOneTensor(self.evaluate() + other)

    def __abs__(self):
        return abs(self.data)

    __radd__ = __add__
    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if is_scalar(other):
            return RankOneTensor(self.evaluate() * other)
        else:
            return self.evaluate() * other
    
    def __truediv__(self, other):
        return self.__mul__(1/other)

    def __rmul__(self, other):
        return self * other
    
    def __str__(self):
        return "{"f"{self.value}""}"

    def __eq__(self, other):
        return self.value == other

class IndexedTensor(_CompositeTensors):
    _tensor_type = ESTensor
    def __init__(self, tensor, *indices):
        self.tensor = tensor
        self.indices = indices
        self.dim = tensor.dim
 
    @property
    def indices_clean(self):
        s = [x[0].replace("/", "")+x[1:] if (type(x) is str) else x for x in self.indices]
        return s

    @property
    def children(self):
        return [self]

        
    def __mul__(self, other, reversed=False):
        if isinstance(other, IndexedTensor):
            # find repeated indices:
            common_index = False
            for a in self.indices_clean:
                if int_like(a): continue
                if a in other.indices_clean: 
                    common_index = a
                    self_index_num = self.indices_clean.index(a)
                    # make it dirty again
                    # common_index = self.indices[self_index_num]
                    index_range = self.tensor.shape[self_index_num]
                    break
                    # other_index_num = other.indices.index(a)
            if not common_index:
                if reversed:
                    self, other = other, self
                return TensorProduct(self, other)
            else:
                if reversed:
                    self, other = other, self
                return TensorSum(
                    *[self.tensor(*list_swap(self.indices, common_index, i)) *
                    other.tensor(*list_swap(other.indices, common_index, i))
                    for i in range(0, index_range)]
                )
        elif isinstance(other, _CompositeTensors):
            if reversed:
                self, other = other, self
            return other * self
        elif is_scalar(other):
            return self._tensor_type(self.tensor.name, self.tensor.data * other)(*self.indices)
        else:
            raise NotImplementedError("Can not multiply with type: " + str(type(other)))

    def is_num(self, **indices):
        if indices:
            temp_indices = [indices[x] if x in indices else x for x in self.indices]
            return np.all(int_like(temp_indices)) and (len(temp_indices) == self.dim)
        else:
            return np.all(int_like(self.indices))
    
    def fully_qualified(self):
        return self.is_num()

    def reindexed(self, *indices):
        """ Return a new indexed tensor with the same data but a new 
        set of indices.
        """
        return (self._tensor_type(self.tensor.name, self.tensor.data, self.tensor.metric)(*indices))

    def evaluate(self, **indices):
        new_indices = list(self.indices)
        new_indices = [indices[x] if x in indices else x for x in self.indices]
        if np.all(int_like(new_indices)):
            return self.tensor.data[tuple(new_indices)]
        if self.is_num():
            # This shouldn't happen?
            print("it happened (bad, IndexedTensor.evaluate())")
            return -1
        else:
            return self.reindexed(*new_indices)

    def __rmul__(self, other):
        if type(other) is type(self):
            return other * self
        else:
            if is_scalar(other):
                return self.__mul__(other,reversed=True)
            else:
                raise TypeError("Can't rmul with non-scalar type {}".format(type(other)))

    def count_matching_indices(self, *indices):
        " Returns the number of given indices that match indices in this Tensor."
        if isinstance(indices[0], IndexedTensor):
            # the method has been called with another tensor
            indices = indices[0].indices_clean
        count = 0
        for i in indices:
            if int_like(i): continue
            if i in self.indices_clean:
                count += 1
        return count
    
    def __add__(self, other):
        if type(other) == type(self):
            return TensorSum(self, other)
        else:
            return (self.tensor + other)(*self.indices)

    def __str__(self):
        return f"{self.tensor}[" + \
            ', '.join([str(x).replace("/", "_") for x in self.indices]) + "]"    

    def expand(self):
        return self


class AbstractESTensor:
    """ Never evaluates as a number, at most a fully indexed label"""
    
    def __init__(self, name, coeff=1, metric=None):
        """ Metric is included only to allow seamless switching with IndexedTensors."""
        self.name = name
        self.coeff = coeff # To be removed
        self.metric = None
    
    @property
    def data(self):
        return self.coeff
    
    def __call__(self, *indices):
        return AbstractIndexedTensor(self, *indices)
    
    def __str__(self):
        return self.name

    def __mul__(self, other):
        if other == 0:
            return 0
        elif is_mutable(other):
            return AbstractESTensor(self.name, coeff=self.coeff*other)

    def __rmul__(self, other):
        return self.__mul__(other)
        # if other == 0:
        #     return 0
        # else:
        #     return TensorProduct(other, self)
    
    

class AbstractIndexedTensor(IndexedTensor):
    _tensor_type = AbstractESTensor
    _blocking = True
    def __init__(self, tensor, *indices) -> None:
        self.name = tensor.name
        self.coeff = tensor.coeff
        self.tensor = tensor
        self.indices = indices


    def evaluate(self, **indices):
        new_indices = list(self.indices)
        new_indices = [indices[x] if x in indices else x for x in self.indices]
        return self.reindexed(*new_indices)
    
    def is_num(self, **indices):
        return False
    
    def fully_qualified(self):
        return np.all(int_like(self.indices))
    
    def __mul__(self, other):
        if type(other) == AbstractIndexedTensor:
            return TensorProduct(self, other)
        elif type(other) == IndexedTensor:
            # The super's __mul__ method doesn't normally work with undefined
            # dimension sizes
            # We can use it anyway and keep the correct multiplication 
            # direction if we set the secret reversed option 
            # in super's __mul__:
            return other.__mul__(self, reversed=True)
        elif is_scalar(other):
            return self._tensor_type(self.name, coeff=self.coeff*other)(*self.indices)
        else:
            return TensorProduct(self, other)
    
    def __rmul__(self, other):
        if type(other) is IndexedTensor:
            return other.__mul__(self, reversed=True)


    def __str__(self):
        ss = super().__str__()
        if self.coeff == 1:
            return ss
        elif self.coeff == -1:
            return "-" + ss
        else:
            return f"{self.coeff}*" + ss


class TensorProduct(_CompositeTensors):
    def __init__(self, *children, sign=1):
        # Distribute any nested products here
        children = list(children)
        self.sign = sign
        numeric = []
        for child in children:
            if type(child) == type(self):
                children.remove(child)
                children += child.children
        for child in children:
            if isinstance(child, _CompositeTensors) and child._blocking:
                break
            if is_scalar(child):
                children.remove(child)
                if child == -1:
                    self.sign *= -1
                elif child != 1:
                    numeric.append(child)
        if numeric:
            children.append(np.prod(numeric))

        self.children = children

        self.permute()

    def permute(self):
        mutables = []
        for child in self.children:
            if isinstance(child, _CompositeTensors) and child._blocking:
                break
            if is_mutable(child):
                mutables.append(child)
                self.children.remove(child)
        if mutables:
            try:
                mutables = [np.prod(mutables)]
                if mutables[0] == 1:
                    return
            except:
                print("Note: Failed to multiply mutable children")
                pass            
            self.children = mutables + self.children

    def expand(self):
        final_sum = TensorSum()

        children = [child for child in self.children]
        changed = False

        if len(children) == 1:
            child = children[0] * self.sign
            if isinstance(child, _CompositeTensors):
                child = child.expand()
            return child

        for i, child_A in enumerate(children):
            if type(child_A) == TensorSum:
                children.remove(child_A)
                product_sum = child_A * TensorProduct(*children)
                final_sum += product_sum
                changed = True
                break
            for j, child_B in enumerate(children):
                if (j-i != 1): continue # Only multiply adjacent items, and from left to right
                # each child is a _CompositeTensor object
                if isinstance(child_A, IndexedTensor):
                    if child_A._blocking and child_A.fully_qualified():
                        # This is probably an operator type term,
                        # leave it intact.
                        continue
                    if isinstance(child_B, IndexedTensor):
                        # See if they can be multiplied
                        if child_A.count_matching_indices(child_B) > 0:
                            # They can be multiplied, expand the multiplication.
                            # Also, remove the old children
                            index_A = children.index(child_A) # This might be different to i
                            index_B = children.index(child_B) # This might be different to j (although should be 1+index_A)
                            children_left = children[:index_A]
                            children_right = children[index_B+1:]
                            # children.remove(child_A)
                            # children.remove(child_B)
                            product = child_A * child_B
                            # The product will be a TensorSum
                            # If there are any more children we now add them
                            # back in.
                            # if children:
                            product_sum = product
                            if children_left:
                                product_sum = TensorProduct(*children_left) * product_sum
                            if children_right:
                                product_sum = product_sum * TensorProduct(*children_right)

                            # else:
                            #     product_sum = product
                            # This will be a sum of products
                            final_sum += product_sum
                            # Done, now ask the sum to expand to continue recursive calls.
                            changed = True
                            break
        if changed:
            final_sum = final_sum.expand()
        else:
            final_sum = self
        self.any_zero()
        if len(final_sum.children) == 1:
            final_sum = final_sum.children[0] * final_sum.sign
        else:
            final_sum.permute()
        return final_sum
      
    def any_zero(self, **indices):
        """ Returns true if any children evaluate to zero with the given indices.
        """
        for child in self.children:
            if is_num(child, **indices):
                if evaluate(child, **indices) == 0:
                    return True
        return False

    def is_num(self, **indices):
        return (np.all([is_num(child, **indices) for child in self.children]) or self.any_zero(**indices))

    def evaluate(self, **indices):
        self.permute()
        if self.any_zero(**indices):
            # If any are zero, the product is just 0.
            # self.children = [0]
            return 0
        elif self.is_num(**indices):
            first = self.children[0].evaluate(**indices)
            for child in self.children[1:]:
                first = first * evaluate(child, **indices)
            return_tensor = first * self.sign
            # This should just be a number
            return return_tensor
        else:
            sign = self.sign
            for child in self.children:
                if evaluate(child, **indices) == -1:
                    sign *= -1
            return_tensor = TensorProduct(*[
                evaluate(child, **indices) for child in self.children if abs(evaluate(child, **indices)) != 1
            ], sign=sign)

            if len(return_tensor.children) == 1:
                return return_tensor.children[0] * return_tensor.sign
            else:
                return_tensor.permute()
                return return_tensor
    
    def _mul(self, other):
        pass 

    def __str__(self):
        if self.sign == -1:
            s = '-'
        else:
            s = ''
        return f"{s}(" + " * ".join([str(x) for x in self.children]) + ")"

    def __repr__(self) -> str:
        return "<TP>" + self.__str__()
    
    def __neg__(self):
        return TensorProduct(*self.children, sign=-self.sign)

    def __mul__(self, other):
        if other == 1:
            return self
        if other == -1:
            return -self
        if isinstance(other, TensorProduct):
            return TensorProduct(*self.children, *other.children, sign=self.sign*other.sign)
        elif isinstance(other, _CompositeTensors):
            if other.num_children == 0:
                return self
            return TensorProduct(self, other, sign=self.sign)
        else:
            try:
                prod = self.children[-1] * other
            except:
                raise NotImplementedError("Unable to multiply the product by type "f"{type(other)}")
            else:
                self.children[-1] = prod
            return self

    def __rmul__(self, other):
        if isinstance(other, _CompositeTensors):
            return other * self 
        # elif is_scalar(other):
        else:
            try:
                prod = other * self.children[0]
            except:
                raise NotImplementedError("Unable to left multiply by type: " + str(type(other)))
            else:
                self.children[0] = prod
            return self

    def __add__(self, other):
        if is_scalar(other) or isinstance(other, _CompositeTensors):
            return TensorSum(self, other)
        elif is_scalar(other) and other == 0:
            return self
        else:
            raise NotImplementedError("Unable to add '"f"{other}""' type " f"{type(other)} to TensorProduct")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

class TensorSum(_CompositeTensors):
    def __init__(self, *children, sign=1):
        children = list(children)
        addable = []
        for child in children:
            if is_scalar(child):
                addable.append(child)
                children.remove(child)
        if addable:
            children.append(sum(addable))
        self.children = [child for child in children if child != 0]
        self.sign = sign
        self.permute()

    def __neg__(self):
        return TensorSum(*self.children, sign=-self.sign)
    

    def permute(self):
        addables = 0
        for child in self.children:
            if isinstance(child, _CompositeTensors) and child._blocking:
                break
            if is_scalar(child):
                addables += child
                self.children.remove(child)
        if addables != 0:
            self.children.append(addables)

    def evaluate(self, **indices):
        self.permute()
        if len(self.children) == 1:
            return self.children[0] * self.sign
        if self.is_num(**indices):
            return self.sign * sum([evaluate(child, **indices) for child in self.children if evaluate(child, **indices) != 0])
        else:
            return TensorSum(*[evaluate(child, **indices) for child in self.children], sign=self.sign)

    # def any_zero(self, **indices):
    #     """ Returns true if any children evaluate to zero with the given indices.
    #     If zeros are found, children are dropped and replaced with a single zero.
    #     """
    #     for child in self.children:
    #         if is_num(child, **indices):
    #             if evaluate(child, **indices) == 0:
    #                 self.children = [0]
    #                 return True
    #     return False


    def is_num(self, **indices):
        # # Check for zeros here
        # for i, child in enumerate(self.children):
        #     if evaluate(child) == 0:
        #         self.children[i] = 0
        return np.all([is_num(child, **indices) for child in self.children])

    def expand(self):
        if len(self.children) == 1:
            child = self.children[0]
            if isinstance(child, _CompositeTensors):
                child = child.expand()
            return child * self.sign
        def _expand(x):
            try:
                return x.expand()
            except:
                return x
        return TensorSum(*[_expand(child) for child in self.children], sign=self.sign)
            
    def __str__(self):
        if self.sign == -1:
            s = '-'
        else:
            s = ''
        return f"{s}(" + " + ".join([str(x) for x in self.children]) + ")"
    
    def __repr__(self):
        return "<TS>" + self.__str__()

    def __mul__(self, other, reversed=False):
        if isinstance(other, _CompositeTensors):
            if other.num_children == 0:
                return self
        
        if other == 1:
            return self
        if other == -1:
            return -self

        if is_scalar(other) or isinstance(other, _CompositeTensors):
            # If reversed, x will flip the product of child and other
            x = -1 if reversed else 1
            return TensorSum(*[TensorProduct(*[child, other][::x]) for child in self.children], sign=self.sign)
        # elif is_scalar(other):
        #     if reversed:
        #         self, other = other, self
        #     return TensorSum()
            # return TensorProduct(self, other)
            # self.children.append(other * self.sign)
            # return self

    def __rmul__(self, other):
        return self.__mul__(other, reversed=True)
    
    def __add__(self, other):
        if isinstance(other, TensorSum):
            if self.sign * other.sign == -1:
                self = self * other.sign
            return TensorSum(*self.children, *other.children, sign=other.sign)
        elif isinstance(other, _CompositeTensors):
            return TensorSum(self, other)
        elif is_scalar(other):
            return TensorSum(*self.children, other)
        else:
            raise NotImplementedError("Addition not defined for Tensors and non-Tensors")
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __iadd__(self, other):
        return self + other

# a = np.array([[1, 2], [3, 4]])
# b = np.array([[4, 2], [0, 0]])
# c = np.array([3, 2])
# Ta = ESTensor("A", a)
# Tb = ESTensor("B", b)
# Tc = ESTensor("c", c)
# D = DifferentialOperator

# x, y, z, t = sympy.symbols("x y z t")
# Dmu = ESTensor("D", [D(t), D(x), D(y), D(z)])

# bigX = ESTensor("X", 
#     [[2*x, x*y, x*z, x*t],
#      [x**2, 4*z, 4*x, 4*y],
#      [2*x, x*y, x*z, x*t],
#      [x**2, 4*z, 4*x, 4*y],
#      ])

# Dmu("mu") * bigX("mu", "nu")



