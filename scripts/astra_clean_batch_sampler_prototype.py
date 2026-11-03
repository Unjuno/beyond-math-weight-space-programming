"""STATIC / NOT IN USE: proposed per-row MPS sampler for later forward service.

This is not a model, does not run a forward, and is never imported by the current
worker. It is not an editor-kernel extension. No equivalence is claimed before a
separately counted forward request checks fresh baseline token sequences.

Semantics: each row owns a saved MPS RNG state initialized with its existing
problem/sample seed. For each step we apply presence -> temperature -> top-k ->
top-p in the same order as current HF generate, restore that row's RNG state,
perform the same one-row torch.multinomial call, and save the resulting state.
The model's native generate loop would be greedy only over these one-hot scores;
sampling itself occurs here. Thus batch row order cannot consume another row's
RNG stream. Batched BF16 model arithmetic can still change logits, so token
equivalence remains an empirical question; this is not an exactness assertion.

Proposed validation after the pending full diagnostic completes: one counted
original-only generation request, same four TRAIN items/seeds, max_tokens=256,
compare all generated token IDs to each fresh sequential diagnostic prefix.
Cost: one of the 50 calls, <=1024 generated tokens. No benchmark access.
"""
import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper


class IndependentMpsSampler(LogitsProcessor):
    def __init__(self, seeds, prompt_length):
        self.prompt_length = prompt_length
        self.states = []
        existing = torch.mps.get_rng_state()
        try:
            for seed in seeds:
                torch.mps.manual_seed(seed)
                self.states.append(torch.mps.get_rng_state().clone())
        finally:
            torch.mps.set_rng_state(existing)
        self.temperature = TemperatureLogitsWarper(.6)
        self.topk = TopKLogitsWarper(20)
        self.topp = TopPLogitsWarper(.95)

    def __call__(self, input_ids, scores):
        assert scores.device.type == 'mps'
        assert len(self.states) == scores.shape[0]
        chosen_scores = torch.full_like(scores, -float('inf'))
        existing = torch.mps.get_rng_state()
        try:
            for i in range(scores.shape[0]):
                row_ids = input_ids[i:i+1]
                row = scores[i:i+1].clone()
                generated = row_ids[0, self.prompt_length:]
                if generated.numel():
                    row[0, torch.unique(generated)] -= 1.5
                row = self.temperature(row_ids, row)
                row = self.topk(row_ids, row)
                row = self.topp(row_ids, row)
                torch.mps.set_rng_state(self.states[i])
                token = torch.multinomial(torch.softmax(row, dim=-1), num_samples=1)
                self.states[i] = torch.mps.get_rng_state().clone()
                chosen_scores[i:i+1].scatter_(1, token, 0.)
        finally:
            torch.mps.set_rng_state(existing)
        return chosen_scores
