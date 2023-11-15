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
    dtype = torch.int16

    def __init__(
            self,
            root: str,
            mode: Literal['SOFTWARE', 'HARDWARE'],
            dim: int,
            num_factors: int,
            num_codevectors: int or Tuple[int], # number of vectors per factor, or tuple of number of codevectors for each factor
            codebooks: str = None,
            fold_dim: int = 256,
            ehd_bits: int = 8, 
            sim_bits: int = 13, 
            seed: None or int = None,  # random seed
            device = None 
        ):

        VSA.mode = mode
        VSA.max_ehd = 2 ** ehd_bits - 1
        VSA.min_ehd = -2 ** ehd_bits
        VSA.max_sim = 2 ** sim_bits - 1
        VSA.min_sim = -2 ** sim_bits
        VSA.fold_dim = fold_dim

        if VSA.mode == "HARDWARE":
            assert(dim % fold_dim == 0)
            # In hardware mode, we use int16 to avoid type conversion for EHD and similarity (which are generally below 16 bits)
            VSA.dtype = torch.int16
        elif VSA.mode == "SOFTWARE":
            # Based on some preliminary experiments, on GPU float32 is faster than int32 because no type conversion needed for matmul (multiset, dot similarity)
            # But float32 slows down prod a lot in CPU. I think it still makes more sense to use int32.
            VSA.dtype = torch.int32
            # VSA.dtype = torch.float32

        self.root = root
        self.dim = dim
        # self.device = device
        self.num_factors = num_factors
        self.num_codevectors = num_codevectors

        if seed is not None:
            torch.manual_seed(seed)

        # Generate codebooks
        if codebooks is not None:
            print(f"Loading codebooks from {codebooks}")
            self.codebooks = torch.load(codebooks)
        else:
            file = os.path.join(self.root, "codebooks.pt")
            if os.path.exists(file):
                print(f"Loading codebooks from {file}")
                self.codebooks = torch.load(file)
            else:
                print("Generating codebooks...", end="")
                self.codebooks = self.gen_codebooks(file)


        if type(self.codebooks) == list:
            for i in range(len(self.codebooks)):
                self.codebooks[i] = self.codebooks[i].to(device)
        else:
            self.codebooks = self.codebooks.to(device)

    def gen_codebooks(self, file) -> List[Tensor] or Tensor:
        # * I removed all device mapping so that the codebooks are generated on CPU. This is because I want everything to be stored
        # * on CPU during generation and saved as such. Otherwise, the data samples will be generated in GPU directly and occupying huge
        # * amout of memory.

        # All factors have the same number of vectors
        if (type(self.num_codevectors) == int):
            if self.mode == "SOFTWARE":
                l = self.random((self.num_factors, self.num_codevectors, self.dim))
            elif self.mode == "HARDWARE":
                # Generate the first fold and generate the rest through CA90
                l = self._gen_full_vector(self.random((self.num_factors, self.num_codevectors, self.fold_dim)))
        # Every factor has a different number of vectors
        else:
            l = []
            for i in range(self.num_factors):
                if self.mode == "SOFTWARE":
                    l.append(self.random((self.num_codevectors[i], self.dim)))
                elif self.mode == "HARDWARE":
                    l.append(self._gen_full_vector(self.random((self.num_codevectors[i], self.fold_dim))))

            try:
                l = torch.stack(l)
            except:
                pass

        os.makedirs(os.path.dirname(file), exist_ok=True)
        torch.save(l, file)
        print("Done. Saved to", file)
        return l

    def cleanup(self, inputs: Tensor, codebooks: Tensor or List[Tensor] = None, abs = False):
        '''
        input: `(b*, f*, d)` :tensor
        codebooks: `(f*, n, d)` :tensor or list
        Return: List[Tuple(int)] of length b, similarity
        `inputs` must be quantized
        '''
        if codebooks == None:
            codebooks = self.codebooks

        if inputs.dim() == 1 and (type(codebooks) == Tensor and codebooks.dim() == 2 or type(codebooks) == list and len(codebooks) == 1 and codebooks[0].dim() == 2):
            # Input is a single vector, so must be a single codebook (tensor)
            winners = torch.empty(len(codebooks), dtype=torch.int64, device=inputs.device)      # gather requires int64 for index
            # Commented out the winner_sims because they are not currently used and gather adds huge overhead especially for uneven codebooks with low batch number
            # winner_sims = torch.empty(len(codebooks), dtype=inputs.dtype, device=inputs.device)
            if type(codebooks) == list:
                for i in range(len(codebooks)):
                    similarities = self.dot_similarity(inputs, codebooks[i])
                    similarities = torch.abs(similarities) if abs else similarities
                    winners[i] = torch.argmax(similarities, -1)
                    # winner_sims[i] = similarities[winners[i]]
            else:
                similarities = self.dot_similarity(inputs, codebooks)
                similarities = torch.abs(similarities) if abs else similarities
                winners = torch.argmax(similarities, -1)
                # winner_sims = similarities.gather(-1, winners)

            return tuple(winners.tolist())
        elif inputs.dim() >= 2 and (type(codebooks) == list and codebooks[0].dim() == 2 or codebooks.dim() == 3):
            # Input is (b, f, d) or (f, d)
            # winners shape = (b, f) or (f) [if b isn't given]
            winners = torch.empty((*inputs.shape[:-2], len(codebooks)), dtype=torch.int64, device=inputs.device)
            # winner_sims = torch.empty((*inputs.shape[:-2], len(codebooks)), dtype=inputs.dtype, device=inputs.device)
            if type(codebooks) == list:
                for i in range(len(codebooks)):
                    similarities = self.dot_similarity(inputs[..., i, :], codebooks[i])
                    similarities = torch.abs(similarities) if abs else similarities
                    winners[...,i] = torch.argmax(similarities, -1)
                    # if similarities.dim() == 2:
                    #     winner_sims[...,i] = similarities.gather(-1, winners[...,i].unsqueeze(-1))[:,0]
                    # else:
                    #     winner_sims[...,i] = similarities.gather(-1, winners[...,i])
            else:
                similarities = self.dot_similarity(inputs, codebooks)
                similarities = torch.abs(similarities) if abs else similarities
                winners = torch.argmax(similarities, -1)
                # winner_sims = similarities.gather(-1, winners.unsqueeze(-1))

            # Innermost dimension is the factor index, must be tuple
            if (winners.dim() > 1):
                winners = [tuple(winners[i].tolist()) for i in range(winners.size(0))]
            else:
                winners = tuple(winners.tolist())
                # winners[..., :]
            return winners
        else:
            raise NotImplementedError("Not implemented for this shape")

      
    def _get_vector(self, key: tuple, codebooks = None, device = None):
        '''
        `key` is a tuple of indices of each factor
        Instead of pre-generate the dictionary, we combine factors to get the vector on the fly
        This saves memory, and also the dictionary lookup is only used during sampling and comparison
        The vector doesn't need to be composed of all available factors. Factors that are None are not bound.
        '''
        if codebooks == None:
            codebooks = self.codebooks
        factors = []
        for i in range(len(key)):
            if key[i] != None:
                factors.append(codebooks[i][key[i]])
        factors = torch.stack(factors)
        return self.multibind(factors).to(device)

    def get_vector(self, key: list or tuple, codebooks = None, quantize = False, device = None):
        '''
        `key` is a list of tuples in [(f0, f1, f2, ...), ...] format, or a single tuple
        fx is the index of the codevector in a codebook, which is also its label.
        When the key is a list, the vectors are bundled into one vector: whether the bundled the result is quantized 
        depends on the `quantize` parameter - even if the list size is 1 (single vector)
        When the key is a tuple, the vector is not bundled hence is automatically quantized.

        Users must be careful not to quantize multiple times when running in hardware mode, which will lead to incorrect results.
        '''
        
        if (type(key) == tuple):
            return self._get_vector(key, codebooks)
        else:
            return self.multiset(torch.stack([self._get_vector(key[i], codebooks) for i in range(len(key))]), quantize=quantize).to(device)

    @classmethod
    def empty(cls, size: tuple or torch.Size or int, dtype=None, device=None) -> Tensor:
        """
            size: (b, n) or (n) or n    (Exclude vector dimension)
            dimensions: d
        """
        # Hardware by default uses int16 in order to support EHD and similarity (which are generally below 16 bits) without type conversion
        if dtype is None:
            dtype = VSA.dtype

        return torch.empty(size, dtype=dtype, device=device)

    @classmethod
    def random(cls, size: torch.Size or tuple or int, dtype=None, device=None) -> Tensor:
        """
            size: (b, n, d) or (n, d) or (d) or d    (Exclude vector dimension)
        """
        if dtype is None:
            dtype = VSA.dtype

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
        if others.nelement() == 0:
            return input.clone()

        if cls.mode == "SOFTWARE":
            return torch.mul(input, others)
        elif cls.mode == "HARDWARE":
            return torch.logical_xor(input, others).to(input.dtype)

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
            result = torch.prod(inputs, dim=-2, dtype=inputs.dtype)
            return result
        elif cls.mode == "HARDWARE":
            n = inputs.size(-2)
            n_ = biggest_power_two(n)
            output = inputs[..., :n_, :]

            # Parallelize many XORs in a hierarchical manner
            # for larger batches this is significantly faster
            while output.size(-2) > 1:
                output = torch.logical_xor(output[..., 0::2, :], output[..., 1::2, :]).to(inputs.dtype)

            output = output.squeeze(-2)

            # TODO: as an optimization we could also perform the hierarchical XOR
            # on the leftovers in a recursive fashion
            leftovers = torch.unbind(inputs[..., n_:, :], -2)
            for i in range(n - n_):
                output = torch.logical_xor(output, leftovers[i]).to(inputs.dtype)

            return output.to(inputs.dtype)

    @classmethod
    def bundle(cls, input: Tensor, others: Tensor, weights_input: Tensor = None, weights_other: Tensor = None, quantize = False) -> Tensor:
        """Currently require inputs to be expanded vectors (doesn't matter for software mode)
        """
        if cls.mode == "SOFTWARE":
            result = torch.add(input, others)
        elif cls.mode == "HARDWARE":
            # Right now we assume the input dtype is wide enough to prevent overflow, which is most likely true since we clip in every operation
            result = torch.add(input, others)
            # Clipping
            result = torch.where(result > VSA.max_ehd, VSA.max_ehd, result)
            result = torch.where(result < VSA.min_ehd, VSA.min_ehd, result)

        if quantize:
            result = cls.quantize(result)

        return result

    @classmethod
    def multiset(cls, inputs: Tensor, weights: Tensor = None, quantize = False) -> Tensor:
        """ Bundle multiple hypervectors
            Currently only support the case when inputs are quantized vectors (doesn't matter for software mode)
            Shape:
                - inputs:   :math:`(b*, n*, v, d)` or :math:`(b*, v, d)`
                - weights: :math:`(b*, n*, v)` or :math:`(b*, v)`
                - output:  :math:`(b*, n*, d)` or :math:`(b*, d)`
        """
        if inputs.dim() < 2:
            raise RuntimeError(
                f"data needs to have at least two dimensions for multiset, got size: {tuple(inputs.shape)}"
        )

        if weights != None:
            assert(inputs.size(-2) == weights.size(-1))

        if cls.mode == "HARDWARE":
            shape = list(inputs.shape)
            del shape[-2]
            result = torch.zeros(shape, dtype=inputs.dtype, device=inputs.device)
            # Use expand and bundle methods so that clipping is taken care of
            if weights != None:
                inputs = cls.expand(inputs, weights)
            else:
                inputs = cls.expand(inputs)

            for i in range(inputs.size(-2)):
                result = cls.bundle(result, inputs[..., i, :])

        elif cls.mode == "SOFTWARE":
        if weights != None:
            # CUDA only supports float32 for matmul
            if inputs.device.type == "cuda":
                result = torch.matmul(weights.unsqueeze(-2).type(torch.float32), inputs.type(torch.float32)).squeeze(-2).type(inputs.dtype)
            else:
                result = torch.matmul(weights.unsqueeze(-2), inputs.squeeze(-2)).type(inputs.dtype)
        else:
            result = torch.sum(inputs, dim=-2, dtype=inputs.dtype) 

        if quantize:
            result = cls.quantize(result)

        return result

    @classmethod
    def dot_similarity(cls, input: Tensor, others: Tensor) -> Tensor:
        """Inner product between hypervectors.
        Input vectors are expected to be quantized
        Shapes:
            - input:   :math:`(b*, n, d)` or :math:`(b*, d)`
                - b is batch [optional], n is the number of vectors to perform comparison [optional]
            - others: :math:`(n*, v*, d)`: 
                - n = vectors [optional], v each of the n vectors in self is compared to v vectors
                - n must match the n in self [optional], v is the number of vectors to compare against each of the n vectors in self [optional]
        """
        if cls.mode == "SOFTWARE":
            if (input.dim() >= 2 and others.dim() == 3):
                assert(others.size(0) == input.size(-2))
                # input is (b*, n, d) and others is (n, v, d)
                # CUDA only supports float32 for matmul 
                if input.device.type == "cuda":
                    result = torch.matmul(input.unsqueeze(-2).type(torch.float32), others.transpose(-2,-1).type(torch.float32)).squeeze(-2).type(input.dtype)
                else:
                    result = torch.matmul(input.unsqueeze(-2), others.transpose(-2,-1)).squeeze(-2)
            elif (input.dim() >= 1 and others.dim() == 2):
                # input is (b*, d) and others is (v, d)
                # CUDA only supports float32 for matmul 
                if input.device.type == "cuda":
                    result = torch.matmul(input.unsqueeze(-2).type(torch.float32), others.transpose(-2,-1).type(torch.float32)).squeeze(-2).type(input.dtype)
                else:
                    result = torch.matmul(input.unsqueeze(-2), others.transpose(-2,-1)).squeeze(-2)
            elif (input.dim() >= 1 and others.dim() == 1):
                # input is (b*, d) and others is (d)
                # CUDA only supports float32 for matmul 
                if input.device.type == "cuda":
                    result = torch.matmul(input.type(torch.float32), others.type(torch.float32)).type(input.dtype)
                else:
                    result = torch.matmul(input, others).type(input.dtype)
            else:
                raise NotImplementedError("Not implemented for this case")

            return result.type(input.dtype)
        elif cls.mode == "HARDWARE":
            positive = torch.tensor(1, dtype=input.dtype, device=input.device)
            negative = torch.tensor(-1, dtype=input.dtype, device=input.device)
            if (input.dim() >= 2 and others.dim() == 3):
                # input is (b*, n, d) and others is (n, v, d)
                assert(others.size(0) == input.size(-2))
                popcount = torch.where(input.unsqueeze(-2) == others, positive, negative)
            elif (input.dim() >= 1 and others.dim() == 2):
                # input is (b*, d) and others is (v, d)
                popcount = torch.where(input.unsqueeze(-2) == others, positive, negative)
            elif (input.dim() >= 1 and others.dim() == 1):
                # input is (b*, d) and others is (d)
                popcount = torch.where(input == others, 1, -1)
            else:
                raise NotImplementedError("Not implemented for this case")
                # popcount = torch.where(input == others, 1, -1)
            
            result = torch.sum(popcount, dim=-1, dtype=input.dtype)
            # Clipping
            result = torch.where(result > VSA.max_sim, VSA.max_sim, result)
            result = torch.where(result < VSA.min_sim, VSA.min_sim, result)

            return result


    @classmethod
    #TODO: hardware mode clipping not done yet
    def hamming_similarity(cls, input: Tensor, others: Tensor) -> Tensor:
        """Hamming similarity between hypervectors.
        Input vectors are expected to be quantized
        Shapes:
            - input:   :math:`(b*, n*, d)`
                - b is batch [optional], n is the number of vectors to perform comparison [optional]
            - others: :math:`(n*, v*, d)`:  n = vectors [optional], v each of the n vectors in self is compared to v vectors
                - n must match the n in self [optional], v is the number of vectors to compare against each of the n vectors in self [optional]
        """
        if (input.dim() >= 2 and others.dim() == 3):
            assert(others.size(0) == input.size(-2))
            # input is (b*, n, d) and others is (n, v, d)
            result = torch.sum(torch.where(input.unsqueeze(-2) == others, 1, 0), dim=-1)
        elif (input.dim() >= 1 and others.dim() == 2):
            # input is (b*, d) and others is (v, d)
            result = torch.sum(torch.where(input.unsqueeze(-2) == others, 1, 0), dim=-1)
        elif (input.dim() >= 1 and others.dim() == 1):
            # input is (b*, d) and others is (d)
            result = torch.sum(torch.where(input == others, 1, 0), dim=-1)
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
            positive = torch.tensor(0, dtype=input.dtype, device=input.device)
            negative = torch.tensor(1, dtype=input.dtype, device=input.device)

        zero = torch.tensor(0, dtype=input.dtype, device=input.device)

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

        result = torch.where(input >= zero, positive, negative)
        return result

    @classmethod
    def expand(cls, input, weight = None) -> Tensor:
        """
        Shape: 
            input:  (b*, n*, v, d) or (b*, v, d) or (d)
            weight: (b*, n*, v) or or (b*, v) or (1)
        """
        if cls.mode == "SOFTWARE":
            if weight is None:
                return input
            else:
                if (input.dim() == 1 and weight.dim() == 1):
                    return input * weight
                else:
                    assert(input.dim() >= 2 and weight.dim() >= 1 and input.size(-2) == weight.size(-1))
                    return input * weight.unsqueeze(-1)
        elif cls.mode == "HARDWARE":
            input = torch.where(input == 0, 1, -1)
            if weight is None:
                return input
            else:
                if (input.dim() == 1 and weight.dim() == 1):
                    result = input * weight
                else:
                    assert(input.dim() >= 2 and weight.dim() >= 1 and input.size(-2) == weight.size(-1))
                    result = input * weight.unsqueeze(-1)

                # Clipping
                result = torch.where(result > VSA.max_ehd, VSA.max_ehd, result)
                result = torch.where(result < VSA.min_ehd, VSA.min_ehd, result)
                return result
    
    @classmethod
    def permute(cls, input: Tensor) -> Tensor:
        """
        Permute the vector
        """
        if cls.mode == "SOFTWARE":
            return input.roll(1, -1)
        elif cls.mode == "HARDWARE":
            # TODO permute per-fold
            return input.roll(1, -1)

    @classmethod
    def is_quantized(cls, input: Tensor) -> bool:
        """For hardware mode, this function may not work correctly because an expanded vector
           may potentially be all 0's and 1's, especially if it underwent subtraction"""
        if cls.mode == "SOFTWARE":
            return torch.logical_or(input == 1, input == -1).all()
        elif cls.mode == "HARDWARE":
            return torch.logical_or(input == 1, input == 0).all()

    @classmethod
    def energy(cls, input: Tensor) -> Tensor:
        """
        The energy in the vector is indicated by the number of non-zero elements
        `input` is expected to be an expanded (unquantized) vector
        """
        return torch.sum(torch.where(input == 0, 0, 1), dim=-1)

    @classmethod
    def ca90(cls, input: Tensor) -> Tensor:
        v1 = input.roll(1, -1)
        v2 = input.roll(-1, -1)
        return torch.logical_xor(v1, v2).to(input.dtype)

    def apply_noise(self, vector: Tensor, noise: float = 0.0, quantized = True) -> Tensor:
        out = vector.clone()
        indices = torch.rand(vector.shape) < noise
        if quantized:
            out[indices] = VSA.inverse(vector[indices])
        else:
            # Not a very good way to apply noise
            out[indices] = torch.neg(vector[indices])
            # 0 is not affected by neg, so we need to manually set it to -1 (normally it quantizes to 1)
            out[indices] = torch.where(out[indices] == 0, -1, out[indices])
        
        return out
    
    def _gen_full_vector(self, fold: Tensor) -> Tensor:
        """
        Generate the rest of the vector through CA90
        """
        assert(fold.size(-1) == self.fold_dim)

        vector = fold.clone()
        for i in range(self.dim // self.fold_dim - 1):
            fold = self.ca90(fold)
            vector = torch.cat((vector, fold), dim=-1)
        return vector