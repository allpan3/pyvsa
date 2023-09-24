from torch import Tensor
from .vsa_tensor import VSATensor
import random as rand
import torch

def random(num_vectors: int, dimensions: int, dtype = None, device = None) -> VSATensor:
    return VSATensor.random(num_vectors, dimensions, dtype, device)

def bind(input: VSATensor, others: VSATensor) -> VSATensor:
    return input.bind(others)

def multibind(inputs: VSATensor) -> VSATensor:
    return inputs.multibind()

def bundle(input: VSATensor, others: VSATensor, quantize = False) -> VSATensor:
    return input.bundle(others, quantize)

def multiset(inputs: VSATensor, weights: Tensor = None, quantize = False) -> VSATensor:
    return inputs.multiset(weights, quantize)

def dot_similarity(input: VSATensor, others: VSATensor) -> Tensor:
    return input.dot_similarity(others)

def apply_noise(vector: VSATensor, quantized = True, noise: float = 0.0) -> VSATensor:
    indices = [rand.random() < noise for i in range(vector.size(-1))]
    if quantized:
        vector[indices] = vector[indices].inverse()
    else:
        # Not a very good way to apply noise
        vector[indices] = torch.neg(vector[indices])
    
    return vector
