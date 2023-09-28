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

import torch
import os.path
from torch import Tensor
from typing import List, Tuple
from typing import Literal

class VSA:

    mode: Literal['SOFTWARE', 'HARDWARE']

    def __init__(
            self,
            root: str,
            mode: Literal['SOFTWARE', 'HARDWARE'],
            dim: int,
            num_factors: int,
            num_codevectors: int or Tuple[int], # number of vectors per factor, or tuple of number of codevectors for each factor
            seed: None or int = None,  # random seed
            device = "cpu"
        ):

        VSA.mode = mode

        self.root = root
        self.dim = dim
        self.device = device
        self.num_factors = num_factors
        self.num_codevectors = num_codevectors

        if seed is not None:
            torch.manual_seed(seed)

        # Generate codebooks
        if self._check_exists("codebooks.pt"):
            self.codebooks = torch.load(os.path.join(self.root, "codebooks.pt"), map_location=self.device)
        else:
            self.codebooks = self.gen_codebooks()

    def gen_codebooks(self) -> List[Tensor] or Tensor:
        l = []
        # All factors have the same number of vectors
        if (type(self.num_codevectors) == int):
            for i in range(self.num_factors):
                l.append(self.random(self.dim, self.num_codevectors, device=self.device))
        # Every factor has a different number of vectors
        else:
            for i in range(self.num_factors):
                l.append(self.random(self.dim, self.num_codevectors[i], device=self.device))

        try:
            l = torch.stack(l).to(self.device)
        except:
            pass

        os.makedirs(self.root, exist_ok=True)
        torch.save(l, os.path.join(self.root, f"codebooks.pt"))

        return l

    def cleanup(self, inputs: Tensor, codebooks: Tensor or List[Tensor] = None, abs = True):
        '''
        input: `(b, f, d)` :tensor. b is batch size, f is number of factors, d is dimension
        Return: List[Tuple(int)] of length b
        `inputs` must be quantized
        '''
        if codebooks == None:
            codebooks = self.codebooks
        
        if type(codebooks) == list:
            winners = torch.empty((inputs.size(0), len(codebooks)), dtype=torch.int32, device=self.device)
            for i in range(len(codebooks)):
                if abs:
                    winners[:,i] = torch.argmax(torch.abs(self.dot_similarity(inputs[:,i], codebooks[i])), -1)
                else:
                    winners[:,i] = torch.argmax(self.dot_similarity(inputs[:,i], codebooks[i]), -1)
            return [tuple(winners[i].tolist()) for i in range(winners.size(0))]
        else:
            if abs:
                winners = torch.argmax(torch.abs(self.dot_similarity(inputs, codebooks)), -1)
            else:
                winners = torch.argmax(self.dot_similarity(inputs, codebooks), -1)
            return [tuple(winners[i].tolist()) for i in range(winners.size(0))]
      
    def _get_vector(self, key:tuple):
        '''
        `key` is a tuple of indices of each factor
        Instead of pre-generate the dictionary, we combine factors to get the vector on the fly
        This saves memory, and also the dictionary lookup is only used during sampling and comparison
        The vector doesn't need to be composed of all available factors. Only the first n codebooks
        are used when the key length is n.
        '''
        factors = torch.stack([self.codebooks[i][key[i]] for i in range(len(key))])
        return self.multibind(factors).to(self.device)

    def get_vector(self, key: list or tuple, quantize = False):
        '''
        `key` is a list of tuples in [(f0, f1, f2, ...), ...] format, or a single tuple
        fx is the index of the codevector in a codebook, which is also its label.
        When the key is a list, the vectors are bundled into one vector: whether the bundled the result is quantized 
        depends on the `quantize` parameter - even if the list size is 1 (single vector)
        When the key is a tuple, the vector is not bundled hence is automatically quantized.

        Users must be careful not to quantize multiple times when running in hardware mode, which will lead to incorrect results.
        '''
        
        if (type(key) == tuple):
            return self._get_vector(key)
        else:
            return self.multiset(torch.stack([self._get_vector(key[i]) for i in range(len(key))]), quantize=quantize)

    def _check_exists(self, file) -> bool:
        return os.path.exists(os.path.join(self.root, file))


    @classmethod
    def empty(cls, num_vectors: int, dimensions: int, dtype=None, device=None) -> Tensor:
        if dtype is None:
            dtype = torch.int8

        return torch.empty(num_vectors, dimensions, dtype=dtype, device=device)

    @classmethod
    def random(cls, dimensions: int, num_vectors: int = None, dtype=None, device=None) -> Tensor:
        if dtype is None:
            dtype = torch.int8
        
        if num_vectors is None:
            size = dimensions
        else:
            size = (num_vectors, dimensions)
        select = torch.empty(size, dtype=torch.bool, device=device)
        select.bernoulli_(generator=None)
        if cls.mode == "SOFTWARE":
            result = torch.where(select, -1, +1).to(dtype=dtype, device=device)
        elif cls.mode == "HARDWARE":
            result = torch.where(select, 0, 1).to(dtype=dtype, device=device)

        return result

    @classmethod
    def bind(cls, input: Tensor, others: Tensor) -> Tensor:
        """
        In hardware mode, caller must make sure self is quantized
        """
        if cls.mode == "SOFTWARE":
            return torch.mul(input, others)
        elif cls.mode == "HARDWARE":
            return torch.logical_not(torch.logical_xor(input, others)).to(input.dtype)

    @classmethod
    def multibind(cls, inputs: Tensor) -> Tensor:
        """Bind multiple hypervectors
           In hardware mode, caller must make sure self is quantized
        """
        if inputs.dim() < 2:
            raise RuntimeError(
                f"data needs to have at least two dimensions for multibind, got size: {tuple(inputs.shape)}"
            )

        def biggest_power_two(n):
            """Returns the biggest power of two <= n"""
            # if n is a power of two simply return it
            if not (n & (n - 1)):
                return n

            # else set only the most significant bit
            return int("1" + (len(bin(n)) - 3) * "0", 2)

        if cls.mode == "SOFTWARE":
            result = torch.prod(inputs, dim=-2)
            return result
        elif cls.mode == "HARDWARE":
            n = inputs.size(-2)
            n_ = biggest_power_two(n)
            output = inputs[..., :n_, :]

            # parallelize many XORs in a hierarchical manner
            # for larger batches this is significantly faster
            while output.size(-2) > 1:
                output = torch.logical_not(torch.logical_xor(output[..., 0::2, :], output[..., 1::2, :])).to(inputs.dtype)

            output = output.squeeze(-2)

            # TODO: as an optimization we could also perform the hierarchical XOR
            # on the leftovers in a recursive fashion
            leftovers = torch.unbind(inputs[..., n_:, :], -2)
            for i in range(n - n_):
                output = torch.logical_not(torch.logical_xor(output, leftovers[i])).to(inputs.dtype)

            return output.to(inputs.dtype)

    # TODO support weight
    @classmethod
    def bundle(cls, input: Tensor, others: Tensor, quantize = False) -> Tensor:
        """Currently, only support the case when inputs are quantized vectors (doesn't matter for software mode)
        """
        if cls.mode == "SOFTWARE":
            result = torch.add(input, others)
        elif cls.mode == "HARDWARE":
            min_one = torch.tensor(-1, dtype=input.dtype, device=input.device)
            _inputs = torch.where(input == 0, min_one, input)
            _others = torch.where(others == 0, min_one, others)

            result = torch.add(_inputs, _others)

        if quantize:
            result = cls.quantize(result)

        return result

    @classmethod
    def multiset(cls, inputs: Tensor, weights: Tensor = None, quantize = False) -> Tensor:
        """ Bundle multiple hypervectors
            Currently only support the case when inputs are quantized vectors (doesn't matter for software mode)
            Shape:
                - self:   :math:`(b*, n*, v, d)`
                - weights: :math:`(b*, n*, v)`
                - output:  :math:`(b*, n*, d)`
        """
        if inputs.dim() < 2:
            raise RuntimeError(
                f"data needs to have at least two dimensions for multiset, got size: {tuple(inputs.shape)}"
        )

        _inputs = inputs
        if cls.mode == "HARDWARE":
            min_one = torch.tensor(-1, dtype=inputs.dtype, device=inputs.device)
            _inputs = torch.where(inputs == 0, min_one, inputs)
 
        if weights != None:
            assert(inputs.size(-2) == weights.size(-1))
            # Add a dimension to weights so that each weight value is applied to all dimensions of the vector
            result = torch.matmul(weights.unsqueeze(-2).type(torch.float32), _inputs.type(torch.float32)).squeeze(-2).type(torch.int64)
        else:
            result = torch.sum(_inputs, dim=-2, dtype=torch.int64) 

        if quantize:
            result = cls.quantize(result)

        return result

    @classmethod
    def dot_similarity(cls, input: Tensor, others: Tensor) -> Tensor:
        """Inner product between hypervectors.
        Input vectors are expected to be quantized
        Shapes:
            - input:   :math:`(b*, n*, d)`
                - b is batch [optional], n is the number of vectors to perform comparison [optional]
            - others: :math:`(n*, v*, d)`:  n = vectors [optional], v each of the n vectors in self is compared to v vectors
                - n must match the n in self [optional], v is the number of vectors to compare against each of the n vectors in self [optional]
        """

        if cls.mode == "SOFTWARE":
            if (input.dim() >= 2 and others.dim() == 3):
                assert(others.size(0) == input.size(-2))
                # input is (b*, n, d) and others is (n, v, d)
                result = torch.matmul(input.unsqueeze(-2).type(torch.float32), others.transpose(-2,-1).type(torch.float32)).squeeze(-2)
            elif (input.dim() >= 1 and others.dim() == 2):
                # input is (b*, d) and others is (v, d)
                result = torch.matmul(input.unsqueeze(-2).type(torch.float32), others.transpose(-2,-1).type(torch.float32)).squeeze(-2)
            elif (input.dim() >= 1 and others.dim() == 1):
                # input is (b*, d) and others is (d)
                result = torch.matmul(input.type(torch.float32), others.type(torch.float32))
            else:
                raise NotImplementedError("Not implemented for this case")

            return result.to(torch.int64)
        elif cls.mode == "HARDWARE":
            if (input.dim() >= 2 and others.dim() == 3):
                # input is (b*, n, d) and others is (n, v, d)
                assert(others.size(0) == input.size(-2))
                popcount = torch.where(input.unsqueeze(-2) == others, 1, -1)
            elif (input.dim() >= 1 and others.dim() == 2):
                # input is (b*, d) and others is (v, d)
                popcount = torch.where(input.unsqueeze(-2) == others, 1, -1)
            elif (input.dim() >= 1 and others.dim() == 1):
                # input is (b*, d) and others is (d)
                popcount = torch.where(input == others, 1, -1)
            else:
                raise NotImplementedError("Not implemented for this case")
                # popcount = torch.where(input == others, 1, -1)

            return torch.sum(popcount, dim=-1, dtype=torch.int64)

    @classmethod
    def hamming_similarity(cls, input: Tensor, others: Tensor) -> Tensor:
        """Hamming similarity between hypervectors.
        Input vectors are expected to be quantized
        Shapes:
            - input:   :math:`(b*, n*, d)`
                - b is batch [optional], n is the number of vectors to perform comparison [optional]
            - others: :math:`(n*, v*, d)`:  n = vectors [optional], v each of the n vectors in self is compared to v vectors
                - n must match the n in self [optional], v is the number of vectors to compare against each of the n vectors in self [optional]
        """
        if cls.mode == "SOFTWARE":
            positive = torch.tensor(1, device=input.device)
            negative = torch.tensor(-1, device=input.device)
            v1 = torch.where(input >= 0, positive, negative)
            v2 = torch.where(others >= 0, positive, negative)
        elif cls.mode == "HARDWARE":
            positive = torch.tensor(1, device=input.device)
            negative = torch.tensor(0, device=input.device)
            v1 = torch.where(input >= 0, positive, negative)
            v2 = torch.where(others >= 0, positive, negative)

        if (input.dim() >= 2 and others.dim() == 3):
            assert(others.size(0) == input.size(-2))
            # input is (b*, n, d) and others is (n, v, d)
            result = torch.sum(torch.where(v1.unsqueeze(-2) == v2, 1, 0), dim=-1)
        elif (input.dim() >= 1 and others.dim() == 2):
            # input is (b*, d) and others is (v, d)
            result = torch.sum(torch.where(v1.unsqueeze(-2) == v2, 1, 0), dim=-1)
        elif (input.dim() >= 1 and others.dim() == 1):
            # input is (b*, d) and others is (d)
            result = torch.sum(torch.where(v1 == v2, 1, 0), dim=-1)
        else:
            raise NotImplementedError("Not implemented for this case") 

        return result

    @classmethod
    def inverse(cls, input: Tensor, quantized = True) -> Tensor:
        if cls.mode == "SOFTWARE":
            result = torch.neg(input)
        elif cls.mode == "HARDWARE":
            if quantized:
                result = 1 - input
            else:
                result = torch.neg(input)

        return result

    @classmethod
    def quantize(cls, input: Tensor) -> Tensor:
        if cls.mode == "SOFTWARE":
            positive = torch.tensor(1, dtype=input.dtype, device=input.device)
            negative = torch.tensor(-1, dtype=input.dtype, device=input.device)
        elif cls.mode == "HARDWARE":
            positive = torch.tensor(1, dtype=input.dtype, device=input.device)
            negative = torch.tensor(0, dtype=input.dtype, device=input.device)

        # So far doesn't seem like it's making a difference
        # if cls.mode == "SOFTWARE":
        #     # Random tiebreaker 
        #     input = input.clone()
        #     select = torch.empty(input[input == 0].shape, dtype=torch.bool, device=input.device)
        #     select.bernoulli_(generator=None)
        #     input[input==0] = torch.where(select, -1, +1).to(dtype=input.dtype, device=input.device)
        # elif cls.mode == "HARDWARE":
        #     # Alternating 1 and -1
        #     input = input.clone()
        #     tie = torch.ones(input[input == 0].shape, dtype=input.dtype, device=input.device)
        #     tie[::2] = -1
        #     input[input==0] = tie

        result = torch.where(input >= 0, positive, negative)
        return result

    @classmethod
    def expand(cls, input):
        if cls.mode == "SOFTWARE":
            return input
        elif cls.mode == "HARDWARE":
            return torch.where(input == 0, -1, input)
    
    @classmethod
    def is_quantized(cls, input: Tensor) -> bool:
        """For hardware mode, this function may not work correctly because an expanded vector
           may potentially be all 0's and 1's, especially if it underwent subtraction"""
        if cls.mode == "SOFTWARE":
            return all(torch.logical_or(input == 1, input == -1).flatten().tolist())
        elif cls.mode == "HARDWARE":
            return all(torch.logical_or(input == 1, input == 0).flatten().tolist())

    @classmethod
    def energy(cls, input: Tensor) -> Tensor:
        """
        The energy in the vector is indicated by the number of non-zero elements
        `input` is expected to be an expanded (unquantized) vector
        """
        return torch.sum(torch.where(input == 0, 0, 1), dim=-1)

    def apply_noise(self, vector: Tensor, noise: float = 0.0, quantized = True) -> Tensor:
        out = vector.clone()
        indices = torch.rand(vector.shape) < noise
        if quantized:
            out[indices] = VSA.inverse(vector[indices])
        else:
            # Not a very good way to apply noise
            out[indices] = torch.neg(vector[indices])
        
        return out