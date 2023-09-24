# %%
import torch.nn as nn
from torch import Tensor
import torch
from typing import Literal, List
from .vsa import VSA

# %%
class Resonator(nn.Module):

    def __init__(self, vsa:VSA, type="CONCURRENT", activation='NONE', iterations=100, argmax_abs=True, lambd = 0, stoch : float = None, early_converge:float = None, device="cpu"):
        super(Resonator, self).__init__()
        self.to(device)

        self.vsa = vsa
        self.device = device
        self.resonator_type = type
        self.iterations = iterations
        self.activation = activation
        self.argmax_abs = argmax_abs
        self.lamdb = lambd
        self.stoch = stoch
        self.early_converge = early_converge
        
    def forward(self, input, init_estimates, codebooks = None, orig_indices: List[int] = None):
        if codebooks == None:
            codebooks = self.vsa.codebooks
        estimates, convergence = self.resonator_network(input, init_estimates, codebooks)
        # outcome: the indices of the codevectors in the codebooks
        outcome = self.vsa.cleanup(estimates, codebooks, self.argmax_abs)
        # Reorder the outcome to the original codebook order
        if (orig_indices != None):
            outcome = [tuple([outcome[j][i] for i in orig_indices]) for j in range(len(outcome))]

        return outcome, convergence

    def resonator_network(self, input: Tensor, init_estimates: Tensor, codebooks: Tensor or List[Tensor]):
        # Must clone, otherwise the original init_estiamtes will be modified
        estimates = init_estimates.clone()
        old_estimates = init_estimates.clone()
        for k in range(self.iterations):
            if (self.resonator_type == "SEQUENTIAL"):
                estimates, max_sim = self.resonator_stage_seq(input, estimates, codebooks, self.activation, self.lamdb, self.stoch)
            elif (self.resonator_type == "CONCURRENT"):
                estimates, max_sim = self.resonator_stage_concur(input, estimates, codebooks, self.activation, self.lamdb, self.stoch)

            if (self.early_converge):
                # TODO we can stop the particular batch if it has been determined to converge, but still can't stop the loop
                # If the similarity value for any factor exceeds the threshold, stop the loop
                if all(torch.max(max_sim, dim=-1)[0] > int(self.vsa.dim * self.early_converge)):
                    break
                # If the similarity of all factors exceed the treshold, stop the loop
                # if all(max_sim.flatten() > int(self.vsa.dim * self.early_converge)):
                #     break
            # TODO this may not be hardware friendly
            # Absolute convergence is signified by identical estimates in consecutive iterations
            # Sometimes RN can enter "bistable" state where estiamtes are flipping polarity every iteration.
            # This is computationally slow
            if all((estimates == old_estimates).flatten()) or all((self.vsa.inverse(estimates) == old_estimates).flatten()):
                break
            old_estimates = estimates.clone()

        return estimates, k 


    def resonator_stage_seq(self,
                            input: Tensor,
                            estimates: Tensor,
                            codebooks: Tensor or List[Tensor],
                            activation: Literal['NONE', 'ABS', 'NONNEG'] = 'NONE',
                            lamdb: float = 0,
                            stoch: float = None):
        
        # Since we only target MAP, inverse of a vector itself

        if input.dim() == 1:
            b = 1
        else:
            b = input.size(0)
        f = estimates.size(-2)
        max_sim = torch.empty((b, f), dtype=torch.int64, device=self.vsa.device)

        for i in range(estimates.size(-2)):
            # Remove the currently processing factor itself
            rolled = estimates.roll(-i, -2)
            inv_estimates = torch.stack([rolled[j][1:] for j in range(estimates.size(0))])
            inv_others = self.vsa.multibind(inv_estimates)
            new_estimates = self.vsa.bind(input, inv_others)

            similarity = self.vsa.similarity(new_estimates, codebooks[i])

            # Apply stochasticity
            if (stoch):
                similarity += (torch.normal(0, self.vsa.dim, similarity.shape) * stoch).type(torch.int64)

            if (activation == 'ABS'):
                similarity = torch.abs(similarity)
            elif (activation == 'THRESHOLD'):
                similarity = torch.nn.Threshold(int(self.vsa.dim * lamdb), 0)(similarity)
            elif (activation == 'HARDSHRINK'):
                similarity = torch.nn.Hardshrink(lambd=int(self.vsa.dim*lamdb))(similarity.type(torch.float32)).type(torch.int64)


            # Dot Product with the respective weights and sum
            # Update the estimate in place
            estimates[:,i] = self.vsa.multiset(codebooks[i], similarity, quantize=True)

            max_sim[:,i] = torch.max(similarity, dim=-1)[0]

        return estimates, max_sim

    def resonator_stage_concur(self,
                               input: Tensor,
                               estimates: Tensor,
                               codebooks: Tensor or List[Tensor],
                               activation: Literal['NONE', 'ABS', 'NONNEG'] = 'NONE',
                               lamdb: float = 0,
                               stoch: float = None):
        '''
        ARGS:
            input: `(*, d)`. d is dimension (b dim is optional)
            estimates: `(b, f, d)`. b is batch size, f is number of factors, d is dimension
        '''
        f = estimates.size(-2)
        if input.dim() == 1:
            b = 1
        else:
            b = input.size(0)
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

        estimates = torch.stack(rolled, dim=-2)

        # First bind all the other estimates together: z * y, x * z, y * z
        inv_others = self.vsa.multibind(estimates)

        # Then unbind all other estimates from the input: s * (x * y), s * (x * z), s * (y * z)
        new_estimates = self.vsa.bind(input.unsqueeze(-2), inv_others)

        if (type(codebooks) == list):
            # f elements, each is tensor of (b, v)
            similarity = [None] * f 
            # Use int64 to ensure no overflow
            output = torch.empty((b, f, d), dtype=torch.int64, device=self.vsa.device) # hmm why do we need int64 here?
            max_sim = torch.empty((b, f), dtype=torch.int64, device=self.vsa.device)
            for i in range(f):
                # All batches, the i-th factor compared with the i-th codebook
                similarity[i] = self.vsa.similarity(new_estimates[:,i], codebooks[i]) 

                # Apply stochasticity
                if (stoch):
                    similarity[i] += (torch.normal(0, self.vsa.dim, similarity[i].shape) * stoch).type(torch.int64)

                if (activation == 'ABS'):
                    similarity[i] = torch.abs(similarity[i])
                elif (activation == 'THRESHOLD'):
                    similarity[i] = torch.nn.Threshold(int(self.vsa.dim * lamdb), 0)(similarity[i])
                elif (activation == 'HARDSHRINK'):
                    similarity[i] = torch.nn.Hardshrink(lambd=int(self.vsa.dim * lamdb))(similarity[i].type(torch.float32)).type(torch.int64)

                # Dot Product with the respective weights and sum
                output[:,i] = self.vsa.multiset(codebooks[i], similarity[i], quantize=True)

                max_sim[:,i] = torch.max(similarity[i], dim=-1)[0]
        else:
            similarity = self.vsa.similarity(new_estimates.unsqueeze(-2), codebooks)

            # Apply stochasticity
            if (stoch):
                similarity += (torch.normal(0, self.vsa.dim, similarity.shape) * stoch).type(torch.int64)

            # Apply activation
            if (activation == 'ABS'):
                similarity = torch.abs(similarity)
            elif (activation == 'THRESHOLD'):
                similarity = torch.nn.Threshold(int(self.vsa.dim * lamdb), 0)(similarity)
            elif (activation == 'HARDSHRINK'):
                similarity = torch.nn.Hardshrink(lambd=int(self.vsa.dim * lamdb))(similarity.type(torch.float32)).type(torch.int64)
            
            # Dot Product with the respective weights and sum
            output = self.vsa.multiset(codebooks, similarity, quantize=True).squeeze(-2)

            max_sim = torch.max(similarity, dim=-1)[0]
        
        return output, max_sim

    def get_init_estimates(self, codebooks = None, batch_size: int = 1, quantize = True) -> Tensor:
        """
        Generate the initial estimates as well as reorder codebooks for the resonator network.
        Seems like initial estimates always benefit from normalization
        """
        if codebooks == None:
            codebooks = self.vsa.codebooks

        if (type(codebooks) == list):
            guesses = [None] * len(codebooks)
            for i in range(len(codebooks)):
                guesses[i] = self.vsa.multiset(codebooks[i])
            init_estimates = torch.stack(guesses).to(self.device)
        else:
            init_estimates = self.vsa.multiset(codebooks).to(self.device)

        if (quantize):
            init_estimates = self.vsa.quantize(init_estimates)
        
        # * If we don't quantize We should technically still assign 0 randomly to -1 or 1
        # * since 0 would erase the similarity (this apply only to software mode)
        
        return init_estimates.unsqueeze(0).repeat(batch_size,1,1)

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
            codebooks = torch.stack(codebooks).to(self.device)
        except:
            pass
            
        return codebooks, indices

# %%
