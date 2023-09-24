
import torch
import os.path
from typing import List, Tuple
from typing import Literal
from .vsa_tensor import VSATensor
from .functional import *

class VSA:

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

        self.root = root
        self.mode = mode
        self.dim = dim
        self.device = device
        self.num_factors = num_factors
        self.num_codevectors = num_codevectors

        VSATensor.set_mode(mode)

        # Generate codebooks
        if self._check_exists("codebooks.pt"):
            self.codebooks = torch.load(os.path.join(self.root, "codebooks.pt"), map_location=self.device)
        else:
            self.codebooks = self.gen_codebooks(seed)

    def gen_codebooks(self, seed) -> List[VSATensor] or VSATensor:
        if seed is not None:
            torch.manual_seed(seed)
        l = []
        # All factors have the same number of vectors
        if (type(self.num_codevectors) == int):
            for i in range(self.num_factors):
                l.append(random(self.num_codevectors, self.dim, device=self.device))
        # Every factor has a different number of vectors
        else:
            for i in range(self.num_factors):
                l.append(random(self.num_codevectors[i], self.dim, device=self.device))

        try:
            l = torch.stack(l).to(self.device)
        except:
            pass

        os.makedirs(self.root, exist_ok=True)
        torch.save(l, os.path.join(self.root, f"codebooks.pt"))

        return l

    def cleanup(self, inputs: VSATensor, codebooks: VSATensor or List[VSATensor] = None, abs = True):
        '''
        input: `(b, f, d)` :tensor. b is batch size, f is number of factors, d is dimension
        Return: List[Tuple(int)] of length b
        '''
        if codebooks == None:
            codebooks = self.codebooks
        
        if type(codebooks) == list:
            winners = torch.empty((inputs.size(0), len(codebooks)), dtype=torch.int32, device=self.device)
            for i in range(len(codebooks)):
                if abs:
                    winners[:,i] = torch.argmax(torch.abs(dot_similarity(inputs[:,i], codebooks[i])), -1)
                else:
                    winners[:,i] = torch.argmax(dot_similarity(inputs[:,i], codebooks[i]), -1)
            return [tuple(winners[i].tolist()) for i in range(winners.size(0))]
        else:
            if abs:
                winners = torch.argmax(torch.abs(dot_similarity(inputs, codebooks)), -1)
            else:
                winners = torch.argmax(dot_similarity(inputs, codebooks), -1)
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
        return multibind(factors).to(self.device)

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
            return multiset(torch.stack([self._get_vector(key[i]) for i in range(len(key))]), quantize=quantize)

    def _check_exists(self, file) -> bool:
        return os.path.exists(os.path.join(self.root, file))
