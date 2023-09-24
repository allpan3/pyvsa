############################################################################################
# This library is inspired by torchhd library, tailored for our project purpose
# We use only the MAP model, but with two variants: software and hardware modes.
# In the software mode, the definitions and operations are exactly the same as MAP model
# In the hardware mode, the representation is BSC-like, but the operations are still MAP-like.
# We use binary in place of bipolar whenever possible to reduce the hardware cost,
# but the bahaviors still closely follow MAP model.
# For example, we use XNOR for binding to get the exact same input-output mapping. We bipolarize the
# values in bundle and the dot similarity operations to get the same results as MAP model.
# The original library is located at:
# https://github.com/hyperdimensional-computing/torchhd.git
############################################################################################

from __future__ import annotations
import torch
from torch import Tensor
from typing import Literal

class VSATensor(Tensor):

    mode: Literal["SOFTWARE", "HARDWARE"]

    @staticmethod
    def __new__(cls, mode, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, mode: Literal["SOFTWARE", "HARDWARE"]):
        if mode is not None:
            VSATensor.mode = mode


    @classmethod
    def set_mode(cls, mode: Literal["SOFTWARE", "HARDWARE"]):
        cls.mode = mode

    @classmethod
    def random(cls, num_vectors: int, dimensions: int, dtype=None, device=None) -> VSATensor:
        if dtype is None:
            dtype = torch.int8

        size = (num_vectors, dimensions)
        select = torch.empty(size, dtype=torch.bool, device=device)
        select.bernoulli_(generator=None)
        if cls.mode == "SOFTWARE":
            result = torch.where(select, -1, +1).to(dtype=dtype, device=device).as_subclass(VSATensor)
        else:
            result = torch.where(select, 0, 1).to(dtype=dtype, device=device).as_subclass(VSATensor)

        return result

    @classmethod
    def empty(cls, num_vectors: int, dimensions: int, dtype=None, device=None) -> VSATensor:
        if dtype is None:
            dtype = torch.int8

        return torch.empty(num_vectors, dimensions, dtype=dtype, device=device).as_subclass(VSATensor)

    def bind(self, others):
        """
        In hardware mode, caller must make sure self is quantized
        """
        if self.mode == "SOFTWARE":
            return torch.mul(self, others)
        elif self.mode == "HARDWARE":
            return torch.logical_not(torch.logical_xor(self, others)).to(self.dtype)

    def multibind(self) -> VSATensor:
        """Bind multiple hypervectors
           In hardware mode, caller must make sure self is quantized
        """
        if self.dim() < 2:
            raise RuntimeError(
                f"data needs to have at least two dimensions for multibind, got size: {tuple(self.shape)}"
            )

        def biggest_power_two(n):
            """Returns the biggest power of two <= n"""
            # if n is a power of two simply return it
            if not (n & (n - 1)):
                return n

            # else set only the most significant bit
            return int("1" + (len(bin(n)) - 3) * "0", 2)

        if self.mode == "SOFTWARE":
            result = torch.prod(self, dim=-2)
            return result
        elif self.mode == "HARDWARE":
            n = self.size(-2)
            n_ = biggest_power_two(n)
            output = self[..., :n_, :]

            # parallelize many XORs in a hierarchical manner
            # for larger batches this is significantly faster
            while output.size(-2) > 1:
                output = torch.logical_not(torch.logical_xor(output[..., 0::2, :], output[..., 1::2, :])).to(self.dtype)

            output = output.squeeze(-2)

            # TODO: as an optimization we could also perform the hierarchical XOR
            # on the leftovers in a recursive fashion
            leftovers = torch.unbind(self[..., n_:, :], -2)
            for i in range(n - n_):
                output = torch.logical_not(torch.logical_xor(output, leftovers[i])).to(self.dtype)

            return output.to(self.dtype)

    # TODO support weight
    def bundle(self, others:VSATensor, quantize = False) -> VSATensor:
        """Currently, only support the case when inputs are quantized vectors (doesn't matter for software mode)
        """
        if self.mode == "SOFTWARE":
            result = torch.add(self, others)
        elif self.mode == "HARDWARE":
            min_one = torch.tensor(-1, dtype=self.dtype, device=self.device)
            _inputs = torch.where(self == 0, min_one, self)
            _others = torch.where(others == 0, min_one, others)

            result = torch.add(_inputs, _others)

        if quantize:
            result = result.quantize()

        return result

    def multiset(self, weights: Tensor = None, quantize = False) -> VSATensor:
        """ Bundle multiple hypervectors
            Currently only support the case when inputs are quantized vectors (doesn't matter for software mode)
            Shape:
                - self:   :math:`(b*, n*, v, d)`
                - weights: :math:`(b*, n*, v)`
                - output:  :math:`(b*, n*, d)`
        """
        if self.dim() < 2:
            raise RuntimeError(
                f"data needs to have at least two dimensions for multiset, got size: {tuple(self.shape)}"
        )

        _inputs = self
        if self.mode == "HARDWARE":
            min_one = torch.tensor(-1, dtype=self.dtype, device=self.device)
            _inputs = torch.where(self == 0, min_one, self)
 
        if weights != None:
            # assert(self.size(-2) == weights.size(-1))
            # Add a dimension to weights so that each weight value is applied to all dimensions of the vector
            result = torch.matmul(weights.unsqueeze(-2).type(torch.float32), _inputs.type(torch.float32)).squeeze(-2).type(torch.int64)
        else:
            result = torch.sum(_inputs, dim=-2, dtype=torch.int64) 

        if quantize:
            result = result.quantize()

        return result

    def dot_similarity(self, others: VSATensor) -> Tensor:
        """Inner product between hypervectors.
        Input vectors are expected to be quantized
        Shapes:
            - self:   :math:`(b*, n*, d)`
                - b is batch [optional], n is the number of vectors to perform comparison [optional]
            - others: :math:`(n*, v, d)`:  n = vectors [optional], v each of the n vectors in self is compared to v vectors
                - n must match the n in self [optional], v is the number of vectors to compare against each of the n vectors in self
 
            If others only contain 1 vector, must unsqueeze before calling this function
        """

        if self.mode == "SOFTWARE":
            if (self.dim() >= 2 and others.dim() == 3):
                # assert(others.size(0) == self.size(-2))
                # self is (b*, n, d) and others is (n, v, d)
                result = torch.matmul(self.unsqueeze(-2).type(torch.float32), others.transpose(-2,-1).type(torch.float32)).squeeze(-2)
            elif (self.dim() >= 1 and others.dim() == 2):
                # self is (b*, d) and others is (v, d)
                result = torch.matmul(self.unsqueeze(-2).type(torch.float32), others.transpose(-2,-1).type(torch.float32)).squeeze(-2)
            else:
                raise NotImplementedError("Not implemented for this case")

            return result.to(torch.int64)
        else:
            if (self.dim() >= 2 and others.dim() == 3):
                # self is (b*, n, d) and others is (n, v, d)
                assert(others.size(0) == self.size(-2))
                popcount = torch.where(self.unsqueeze(-2) == others, 1, -1)
            elif (self.dim() >= 1 and others.dim() == 2):
                # self is (b*, d) and others is (v, d)
                popcount = torch.where(self.unsqueeze(-2) == others, 1, -1)
            else:
                raise NotImplementedError("Not implemented for this case")
                # popcount = torch.where(self == others, 1, -1)

            return torch.sum(popcount, dim=-1, dtype=torch.int64)


    def inverse(self, quantized = True) -> VSATensor:
        if self.mode == "SOFTWARE":
            result = torch.neg(self)
        elif self.mode == "HARDWARE":
            if quantized:
                result = 1 - self
            else:
                result = torch.neg(self)

        return result

    def quantized(self):
        if self.mode == "SOFTWARE":
            return all(torch.logical_or(self == 1, self == -1).flatten())
        elif self.mode == "HARDWARE":
            # This should guarantee that the vector is quantized, may need to think closer
            return all(torch.logical_or(self == 1, self == 0).flatten())

    def quantize(self):
        if self.mode == "SOFTWARE":
            positive = torch.tensor(1, dtype=self.dtype, device=self.device)
            negative = torch.tensor(-1, dtype=self.dtype, device=self.device)
        elif self.mode == "HARDWARE":
            positive = torch.tensor(1, dtype=self.dtype, device=self.device)
            negative = torch.tensor(0, dtype=self.dtype, device=self.device)

        result = torch.where(self >= 0, positive, negative).as_subclass(VSATensor)
        return result

    def expand(self):
        if self.mode == "SOFTWARE":
            return self
        elif self.mode == "HARDWARE":
            return torch.where(self == 0, -1, self)