from __future__ import annotations
from collections import defaultdict
import torch
import tqdm 
import math
import pdb

from models.wllm import WatermarkDetector, WatermarkLogitsProcessor


class SweetCodeMixin:
    def _get_greenlist_ids(self, input_ids, scores) -> list[int]:
        
        self._seed_rng(input_ids, self.hash_key)
        
        with torch.no_grad():
            # scores = torch.randn(scores.size(), device=scores.device, dtype=scores.dtype, generator=self.rng)
            sorted_values, sorted_indices = torch.sort(scores, descending=True)
            seq_len = sorted_indices.shape[0]
            k = (seq_len + 1) // 2  # Ceiling division to handle odd length
            
            # Generate random choices for each pair and possibly the last element
            random_choices = torch.randint(0, 2, (k,), generator=self.rng, device=sorted_indices.device)
            
            # Create indices for pairs
            base_indices = torch.arange(0, seq_len-1, 2, device=sorted_indices.device)
            selected_indices = base_indices + random_choices[:len(base_indices)]
            
            # If sequence length is odd, potentially include the last element
            if seq_len % 2 == 1:
                if random_choices[-1].item():  # Use the last random choice for the last element
                    selected_indices = torch.cat([selected_indices, sorted_indices.new_tensor([seq_len-1])])
            
            greenlist_ids = sorted_indices[selected_indices]
            
            # Convert to list format for return
            greenlist_ids = greenlist_ids.cpu().tolist()
        
        return greenlist_ids


class SweetCodeLogitsProcessor(SweetCodeMixin, WatermarkLogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx],scores[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        # get entropy
        raw_probs = torch.softmax(scores, dim=-1)  # batch_size, vocab_size
        ent = -torch.where(raw_probs > 0, raw_probs * raw_probs.log(), raw_probs.new([0.0])).sum(dim=-1)
        entropy_mask = (ent > self.entropy_threshold).view(-1, 1)
        
        green_tokens_mask = green_tokens_mask * entropy_mask

        scores = self._bias_greenlist_logits(
            scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta
        )
        return scores
        
class SweetCodeDetector(SweetCodeMixin, WatermarkDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # entropy threshold 포함됨

    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        prefix_len: int,
        entropy: list[float],
        scores: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_watermarking_fraction: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        score_dict = dict()
        prefix_len = max(self.min_prefix_len, prefix_len)
        device = scores.device
        
        self.rng = torch.Generator(device=device)
        
        if self.ignore_repeated_bigrams:
            raise NotImplementedError("not implemented for entropy")
        
        num_tokens_generated = len(input_ids) - prefix_len
        if num_tokens_generated < 1:
            print(f"only {num_tokens_generated} generated : cannot score.")
            score_dict["invalid"] = True
            return score_dict

        try:
            assert len(entropy) == len(input_ids) # eos id나 pad id 있으면 어쩌지?
        except AssertionError:
            print("len(entropy) != len(input_ids)")
            pdb.set_trace()

        num_tokens_scored = num_tokens_generated - len(
            [e for e in entropy[prefix_len:] if e <= self.entropy_threshold]
        )  # entropy_threshold보다 작은 entropy를 가진 token은 score하지 않음.
        if num_tokens_scored < 1:
            assert num_tokens_scored == 0
            # regarding as human generated
            return {
                "num_tokens_generated": num_tokens_generated,
                "num_tokens_scored": 0,
                "num_green_tokens": 0,
                "watermarking_fraction": 0,
                "green_fraction": 0,
                "z_score": 0.0,
                "p_value": 1,
            }

        # Standard method.
        # Since we generally need at least 1 token (for the simplest scheme)
        # we start the iteration over the token sequence with a minimum
        # num tokens as the first prefix for the seeding scheme,
        # and at each step, compute the greenlist induced by the
        # current prefix and check if the current token falls in the greenlist.
        green_token_count, green_token_mask = 0, []
        for idx in range(prefix_len, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self._get_greenlist_ids(input_ids[:idx], scores[idx])

            if entropy[idx] > self.entropy_threshold:
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)
            else:
                # when entropy is low; i.e., watermarking is not applied
                green_token_mask.append(False)

        score_dict.update(dict(num_tokens_generated=num_tokens_generated))
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_watermarking_fraction:
            score_dict.update(
                dict(watermarking_fraction=(num_tokens_scored / num_tokens_generated))
            )
        if return_green_fraction:
            score_dict.update(
                dict(green_fraction=(green_token_count / num_tokens_scored))
            )
        if return_z_score:
            score_dict.update(
                dict(
                    z_score=self._compute_z_score(green_token_count, num_tokens_scored)
                )
            )
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict