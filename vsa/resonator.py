import torch.nn as nn
import torch
from torch import Tensor
from typing import Literal, List
from .vsa import VSA
import math

class Resonator(nn.Module):

    noise = None
    mode : Literal["SOFTWARE", "HARDWARE"] = None

    def __init__(self, vsa:VSA, mode: Literal["SOFTWARE", "HARDWARE"], type="CONCURRENT", activation='NONE', iterations=1000, argmax_abs=True, act_val = None, stoch = "NONE", randomness : float = 0.0, early_converge:float = None, seed = None, device="cpu"):
        super(Resonator, self).__init__()
        self.to(device)
        
        Resonator.mode = mode
        self.vsa = vsa
        self.device = device
        self.resonator_type = type
        self.iterations = iterations
        self.activation = activation
        self.argmax_abs = argmax_abs
        self.act_val = act_val
        self.stoch = stoch
        self.randomness = randomness
        self.early_converge = early_converge
        
        if seed != None:
            torch.manual_seed(seed)

    def forward(self, input: Tensor, init_estimates: Tensor, codebooks = None, known: tuple = None, orig_indices: List[int] = None):
        """
            known: A tuple of indices of factors. Elements whose values are not None are known factors and the value is the index of the codevector in the orginal codebook
                   `known` should only be supplied when codebooks is also supplied  
                   The same `known` is used by all batches
        """
        with torch.profiler.record_function("rn_core"):
            # If known is given, unbind the known factors from input and run resonator network with the rest
            if known is not None:
                assert(codebooks is not None)
                assert(len(known) == len(self.vsa.codebooks))
                known_vector = torch.stack([self.vsa.codebooks[i][known[i]] for i in range(len(known)) if known[i] is not None])
                known_vector = self.vsa.multibind(known_vector)
                input = self.vsa.bind(input, known_vector)

            if codebooks is None:
                codebooks = self.vsa.codebooks

            # Pre-generate a set of noise tensors; had to put it here to accomondate the case where partial codebooks are used
            if self.mode == "HARDWARE" and Resonator.noise == None:
                if (self.stoch == "SIMILARITY"):
                    Resonator.noise = (torch.normal(0, input.size(-1), (1659,), device=self.device) * self.randomness).type(torch.int16)
                    assert(len(Resonator.noise) > sum([codebooks[i].size(0) for i in range(len(codebooks))]))

                elif (self.stoch == "VECTOR"):
                    Resonator.noise = torch.rand(200, input.size(-1), device=self.device) < self.randomness
                    # To mimic hardware, retrict the number of noise vectors we store in memory. The more we store the closer it is to a true random model.
                    # The minimum required is the number of codevectors in the longest codebook so that in each iteration each codevector is applied with different noise
                    assert(Resonator.noise.size(0) >= max([len(codebooks[i]) for i in range(len(codebooks))]))

            estimates, iter, converge = self.resonator_network(input, init_estimates, codebooks)

            # outcome: the indices of the codevectors in the codebooks
            outcome  = self.vsa.cleanup(estimates, codebooks, self.argmax_abs)

            # Insert the known vectors into the outcome
            if known is not None:
                if input.dim() == 1:
                    _outcome = list(known)
                    k = 0
                    for j in range(len(_outcome)):
                        if _outcome[j] is None:
                            _outcome[j] = outcome[k]
                            k += 1
                    outcome = tuple(_outcome)
                else:
                    # per-batch
                    for i in range(len(outcome)):
                        _outcome = list(known)
                        k = 0
                        for j in range(len(_outcome)):
                            if _outcome[j] is None:
                                _outcome[j] = outcome[i][k]
                                k += 1
                        outcome[i] = tuple(_outcome)

            # Reorder the outcome to the original codebook order
            if (orig_indices != None):
                outcome = [tuple([outcome[j][i] for i in orig_indices]) for j in range(len(outcome))]

        return outcome, iter, converge

    def resonator_network(self, input: Tensor, init_estimates: Tensor, codebooks: Tensor or List[Tensor]):
        # Must clone, otherwise the original init_estiamtes will be modified
        estimates = init_estimates.clone()
        old_estimates = init_estimates.clone()
        # TODO separate status for each batch
        converge_status = False
        for k in range(self.iterations):
            with torch.profiler.record_function("rn_iter"):
                if (self.resonator_type == "SEQUENTIAL"):
                    estimates, max_sim = self.resonator_stage_seq(input, estimates, codebooks, self.activation, self.act_val, self.stoch, self.randomness)
                elif (self.resonator_type == "CONCURRENT"):
                    estimates, max_sim = self.resonator_stage_concur(input, estimates, codebooks, self.activation, self.act_val, self.stoch, self.randomness)

                if (self.early_converge):
                    # TODO make this a config option
                    # If the similarity value for any factor exceeds the threshold, stop the loop
                    early_converge = (torch.max(max_sim, dim=-1)[0] > int(input.size(-1) * self.early_converge)).all()
                    if early_converge == True:
                        converge_status = "EARLY"
                        break
                    # If the similarity of all factors exceed the treshold, stop the loop
                    # if (max_sim.flatten() > int(input.size(-1) * self.early_converge)).all() == True:
                    #     break
                # Absolute convergence is signified by identical estimates in consecutive iterations
                # Sometimes RN can enter "bistable" state where estiamtes are flipping polarity every iteration.
                if (estimates == old_estimates).all() == True or (VSA.inverse(estimates) == old_estimates).all() == True:
                    converge_status = "CONVERGED"
                    break
                old_estimates = estimates.clone()
            # TODO we can stop iteration count for a particular batch if it has been determined to converge, but still can't stop the loop
            # That way we can get more accurate iteration count
        return estimates, k+1, converge_status


    def resonator_stage_seq(self,
                            input: Tensor,
                            estimates: Tensor,
                            codebooks: Tensor or List[Tensor],
                            activation = 'IDENTITY',
                            act_val =None,
                            stoch = "NONE", 
                            randomness: float = 0.0) -> Tensor:
        """
        ARGS:
            input: `(b*, d)`
            estimates: (b*, f, d)     
        """
        # Since we only target MAP, inverse of a vector itself

        f = estimates.size(-2)
        if input.dim() == 1:
            # No batch
            assert(estimates.dim() == 2)
            b = 1
            # dot_similarity result type is the same as input
            max_sim = torch.empty(f, dtype=input.dtype, device=self.device) 
        else:
            assert(input.size(0) == estimates.size(0))
            b = input.size(0)
            max_sim = torch.empty((b, f), dtype=input.dtype, device=self.device) 

        for i in range(estimates.size(-2)):
            with torch.profiler.record_function("unbind"):
                # Remove the currently processing factor itself
                rolled = estimates.roll(-i, -2)
                inv_estimates = rolled[..., 1:, :]

                inv_others = VSA.multibind(inv_estimates)
                new_estimates = VSA.bind(input, inv_others)

            _codebook = codebooks[i]
            # Apply to codebook vectors (want to apply different noise for each estimate-codevector comparison, this is easier to code)
            if (stoch == "VECTOR"):
                _codebook = _codebook.clone()
                if (self.mode == "SOFTWARE"):
                    indices = torch.rand(_codebook.shape) < randomness
                    _codebook[indices] = VSA.inverse(_codebook[indices])
                elif (self.mode == "HARDWARE"):
                    _codebook[Resonator.noise[0:_codebook.size(0)]] = VSA.inverse(_codebook[Resonator.noise[0:_codebook.size(0)]])
                    Resonator.noise = Resonator.noise.roll(-_codebook.size(0), 0)
            # similarity.shape = (b, v)

            with torch.profiler.record_function("similarity"):
                similarity = VSA.dot_similarity(new_estimates, _codebook)

            with torch.profiler.record_function("activation"):
                # Apply stochasticity
                if (stoch == "SIMILARITY"):
                    if (self.mode == "SOFTWARE"):
                        similarity += (torch.normal(0, input.size(-1), similarity.shape) * randomness).to(self.device).type(input.dtype)
                    elif (self.mode == "HARDWARE"):
                        similarity += Resonator.noise[0:_codebook.size(0)]
                        Resonator.noise = Resonator.noise.roll(-_codebook.size(0), -1)

                # Apply activation
                if (activation == 'THRESHOLD'):
                    # Wipe out all values below the threshold. Values equal to threshold should stay intact, the same way how we do in hardware
                    similarity = torch.nn.Threshold(act_val-1, 0)(similarity)
                elif (activation == 'SCALEDOWN'):
                    # Sacle down has the similar effect of hardshrink as small values are even smaller. It scales down the large values, which doesn't matter
                    # Note that early convergence threshold value also needs to be scaled accordingly
                    if self.mode == "SOFTWARE":
                        similarity = similarity // act_val
                    elif self.mode == "HARDWARE":
                        # In hardware mode, we use right shift so small positive values are zeroed, and small negative values are pushed to -1
                        # Scaling the value down helps with clipping. Make sure the early convergence threshold accounts for this scaling as well
                        shift = round(math.log2(act_val))
                        similarity = similarity >> shift
                elif (activation == "THRESH_AND_SCALE"):
                    # Combine thresholding and scaling: all negative values are automatically wiped out, then the remaining values are scaled down
                    # Note that early convergence threshold value also needs to be scaled accordingly
                    if self.mode == "SOFTWARE":
                        similarity = torch.nn.Threshold(0, 0)(similarity) // act_val
                    elif self.mode == "HARDWARE":
                        # Note in hardware mode positive values are effectively thresholded by the scale factor, as small values are wiped out by right shift
                        shift = round(math.log2(act_val))
                        similarity = torch.nn.Threshold(0, 0)(similarity) >> shift

            # Dot Product with the respective weights and sum, unsqueeze codebook to account for batch
            # Update the estimate in place
            if estimates.dim() == 3:
                with torch.profiler.record_function("weighted_bundle"):
                    # repeat is slow, but even if we split batch there's still un-utilized time in weighted_bundle. It also sometimes slows down unbinding
                    estimates[:,i] = VSA.multiset(codebooks[i].unsqueeze(0).repeat(b,1,1), similarity, quantize=True)
                    # for j in range(b):
                    #     estimates[j,i] = VSA.multiset(codebooks[i], similarity[j], quantize=True)
                max_sim[:,i] = torch.max(similarity, dim=-1)[0]
            else:
                # No batch
                estimates[i] = VSA.multiset(codebooks[i], similarity, quantize=True)
                max_sim[i] = torch.max(similarity, dim=-1)[0]

        return estimates, max_sim

    def resonator_stage_concur(self,
                               input: Tensor,
                               estimates: Tensor,
                               codebooks: Tensor or List[Tensor],
                               activation = "IDENTITY",
                               act_val = None,
                               stoch = "NONE",
                               randomness: float = 0.0) -> Tensor:
        '''
        ARGS:
            input: `(b*, d)`. d is dimension (b dim is optional)
            estimates: `(b, f, d)`. b is batch size, f is number of factors, d is dimension
        '''

        f = estimates.size(-2)
        if input.dim() == 1:
            # No batch
            assert(estimates.dim() == 2)
            b = 1
            max_sim = torch.empty(f, dtype=input.dtype, device=self.device) 
        else:
            assert(input.size(0) == estimates.size(0))
            b = input.size(0)
            max_sim = torch.empty((b, f), dtype=input.dtype, device=self.device) 
        d = input.size(-1)

        # Since we only target MAP, inverse of a vector itself

        # Roll over the number of estimates to align each row with the other symbols
        # Example: for factorizing x, y, z the stacked matrix has the following estimates:
        # [[z, y],
        #  [x, z],
        #  [y, x]]
        rolled = []
        for i in range(1, f):
            rolled.append(estimates.roll(i, -2))

        if rolled:
            inv_estimates = torch.stack(rolled, dim=-2)
        else:
            # rolled is empty, i.e. only one factor
            inv_estimates = torch.empty((*estimates.shape[:-2], 0, d))

        # First bind all the other estimates together: z * y, x * z, y * z
        inv_others = VSA.multibind(inv_estimates)

        # Then unbind all other estimates from the input: s * (x * y), s * (x * z), s * (y * z)
        new_estimates = VSA.bind(input.unsqueeze(-2), inv_others)

        _codebooks = codebooks
        # Apply to codebook vectors (want to apply different noise for each estimate-codevector comparison, this is easier to code)
        if (stoch == "VECTOR"):
            _codebooks = _codebooks.clone()
            for i in range(f):
                if (self.mode == "SOFTWARE"):
                    indices = torch.rand(_codebooks[i].shape) < randomness
                    _codebooks[i][indices] = VSA.inverse(_codebooks[i][indices])
                elif (self.mode == "HARDWARE"):
                    _codebooks[i][Resonator.noise[0:_codebooks[i].size(0)]] = VSA.inverse(_codebooks[i][Resonator.noise[0:_codebooks[i].size(0)]])
                    Resonator.noise = Resonator.noise.roll(-_codebooks[i].size(0), 0)

        if (type(codebooks) == list):
            # f elements, each is tensor of (b, v)
            similarity = [None] * f
            for i in range(f):
                # All batches, the i-th factor compared with the i-th codebook
                similarity[i] = VSA.dot_similarity(new_estimates[...,i,:], _codebooks[i]) 

                # Apply stochasticity
                if (stoch == "SIMILARITY"):
                    if (self.mode == "SOFTWARE"):
                        similarity[i] += (torch.normal(0, input.size(-1), similarity[i].shape, device=input.device) * randomness).type(input.dtype)
                    elif (self.mode == "HARDWARE"):
                        similarity[i] += Resonator.noise[0:_codebooks[i].size(0)]
                        Resonator.noise = Resonator.noise.roll(-_codebooks[i].size(0), -1)

                # Apply activation; see resonator_stage_seq for detailed comments
                if (activation == 'THRESHOLD'):
                    similarity[i] = torch.nn.Threshold(act_val-1, 0)(similarity[i])
                elif (activation == 'SCALEDOWN'):
                    if self.mode == "SOFTWARE":
                        similarity[i] = similarity[i] // act_val
                    elif self.mode == "HARDWARE":
                        shift = round(math.log2(act_val))
                        similarity[i] = similarity[i] >> shift
                elif (activation == "THRESH_AND_SCALE"):
                    if self.mode == "SOFTWARE":
                        similarity[i] = torch.nn.Threshold(0, 0)(similarity[i]) // act_val
                    elif self.mode == "HARDWARE":
                        shift = round(math.log2(act_val))
                        similarity[i] = torch.nn.Threshold(0, 0)(similarity[i]) >> shift
                
                # Dot Product with the respective weights and sum
                estimates[...,i,:] = VSA.multiset(codebooks[i], similarity[i], quantize=True)
                max_sim[...,i] = torch.max(similarity[i], dim=-1)[0]
        else:
            similarity = VSA.dot_similarity(new_estimates, _codebooks)

            # Apply stochasticity
            if (stoch == "SIMILARITY"):
                if (self.mode == "SOFTWARE"):
                    similarity += (torch.normal(0, input.size(-1), similarity.shape, device=self.device) * randomness).type(input.dtype)
                elif (self.mode == "HARDWARE"):
                    similarity += Resonator.noise[0:_codebooks.size(1)*f].view(f, _codebooks.size(1))
                    Resonator.noise = Resonator.noise.roll(-_codebooks.size(1)*f, -1)

            # Apply activation; see resonator_stage_seq for detailed comments
            if (activation == 'THRESHOLD'):
                similarity = torch.nn.Threshold(act_val-1, 0)(similarity)
            elif (activation == 'SCALEDOWN'):
                if self.mode == "SOFTWARE":
                    similarity = similarity // act_val
                elif self.mode == "HARDWARE":
                    shift = round(math.log2(act_val))
                    similarity = similarity >> shift
            elif (activation == "THRESH_AND_SCALE"):
                if self.mode == "SOFTWARE":
                    similarity = torch.nn.Threshold(0, 0)(similarity) // act_val
                elif self.mode == "HARDWARE":
                    shift = round(math.log2(act_val))
                    similarity = torch.nn.Threshold(0, 0)(similarity) >> shift

            # Dot Product with the respective weights and sum
            estimates = VSA.multiset(codebooks, similarity, quantize=True)
            max_sim = torch.max(similarity, dim=-1)[0]
        
        return estimates, max_sim

    def get_init_estimates(self, codebooks = None, quantize = True) -> Tensor:
        """
        Generate the initial estimates as well as reorder codebooks for the resonator network.
        Seems like initial estimates always benefit from normalization
        """
        if codebooks == None:
            codebooks = self.vsa.codebooks

        if (type(codebooks) == list):
            guesses = [None] * len(codebooks)
            for i in range(len(codebooks)):
                guesses[i] = VSA.multiset(codebooks[i])
            init_estimates = torch.stack(guesses)
        else:
            init_estimates = VSA.multiset(codebooks)

        if (quantize):
            init_estimates = VSA.quantize(init_estimates)
        
        # * If we don't quantize We should technically still assign 0 randomly to -1 or 1
        # * since 0 would erase the similarity (this apply only to software mode)
        
        return init_estimates

    def reorder_codebooks(self, orig_codebooks: List[Tensor] or Tensor = None) -> (List[Tensor] or Tensor, List[int]):
        """
        RETURNS:
            codebooks: reordered codebooks
            indices: the original indices of the new codebooks
        """
        if orig_codebooks == None:
            orig_codebooks = self.vsa.codebooks
        # Sort the codebooks by length, so that the longest codebook is processed first
        # Experiments show that this produces the best results for sequential resonator, and doesn't affect concurrent resonator
        codebooks = sorted(orig_codebooks, key=len, reverse=True)
        # Remember the original indice of the codebooks for reordering later
        indices = sorted(range(len(orig_codebooks)), key=lambda k: len(orig_codebooks[k]), reverse=True)
        indices = [indices.index(i) for i in range(len(indices))]
        try:
            codebooks = torch.stack(codebooks)
        except:
            pass
            
        return codebooks, indices
